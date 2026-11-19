"""Tests for the weight edit, on a tiny random model (CPU, offline). Skipped when torch is not installed.

Run with:  uv run --extra gpu pytest tests/test_edit.py
"""

import json
import math
import warnings

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

from bench import edit as edit_module  # noqa: E402
from bench.edit import (  # noqa: E402
    LADDER,
    Edit,
    applied_change,
    apply_edit,
    compute_targeted_edit,
    default_edit_layer,
    edited,
    find_down_proj,
    letter_token_ids,
    load_edit,
    random_edit_like,
    remove_edit,
    save_edit,
    save_edited_model,
    spectral_norm,
)
from bench.prompts import build_prompt, render_prompt  # noqa: E402
from bench.records import Question  # noqa: E402

LAYER = 2
WIDTH, MLP_WIDTH, N_LAYERS = 32, 64, 4

QUESTION = Question(
    id="2025-10-19/p3/ORD/1", exam_date="2025-10-19", exam_pass=3, section="ORD", number=1,
    question="snarlik", options={"A": "olik", "B": "liknande", "C": "snabb", "D": "sen", "E": "lugn"},
    answer="A",
)


class TinyBackend:
    """Stands in for the HF backend: same attributes, same letter-probability rule, on CPU."""

    def __init__(self, model, tokenizer, tokenizer_path):
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path
        self._letter_ids = letter_token_ids(tokenizer)

    def letter_probabilities(self, prompt, letters, question):
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, logits_to_keep=1).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1)
        return {letter: float(probs[self._letter_ids[letter]].sum()) for letter in letters}


@pytest.fixture(scope="module")
def tokenizer_dir(tmp_path_factory):
    """A word-level tokenizer that knows the words of the test prompt, with a chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    body = build_prompt(QUESTION)[0]["content"]
    words = [word for word, _ in pre_tokenizers.Whitespace().pre_tokenize_str(body + " user assistant :")]
    vocab = {"[UNK]": 0}
    for word in [*words, "A", "B", "C", "D", "E"]:
        vocab.setdefault(word, len(vocab))
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}user : {{ m['content'] }} {% endfor %}assistant :"
    path = tmp_path_factory.mktemp("tokenizer")
    tokenizer.save_pretrained(str(path))
    return path


def make_model(vocab_size, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size, hidden_size=WIDTH, intermediate_size=MLP_WIDTH, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
    )
    return transformers.LlamaForCausalLM(config).to(dtype).eval()


@pytest.fixture()
def backend(tokenizer_dir):
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return TinyBackend(make_model(len(tokenizer)), tokenizer, tokenizer_dir)


def wrong_question(backend, rank):
    """QUESTION with the answer key set to the letter the model ranks `rank`-th (0 = its own choice)."""
    prompt = render_prompt(build_prompt(QUESTION), str(backend.tokenizer_path))
    probs = backend.letter_probabilities(prompt, sorted(QUESTION.options), QUESTION)
    ranked = sorted(probs, key=probs.get, reverse=True)
    return Question(**{**QUESTION.__dict__, "answer": ranked[rank]})


def some_edit(model, alpha=0.5, seed=1):
    name, weight = find_down_proj(model, LAYER)
    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(weight.shape[0], generator=generator)
    v = torch.randn(weight.shape[1], generator=generator)
    return Edit(layer=LAYER, param_name=name, u=u / u.norm(), v=v / v.norm(), alpha=alpha,
                alpha_fraction=0.1, kind="targeted", question_id="q", seed=None)


# ---------------------------------------------------------------- finding the matrix

def test_default_edit_layer_is_two_thirds_up():
    assert default_edit_layer(32) == 21   # Qwen3.5-9B
    assert default_edit_layer(34) == 23   # Gemma 3 4B


def test_find_down_proj_returns_the_named_parameter(backend):
    name, weight = find_down_proj(backend.model, LAYER)
    assert name == f"model.layers.{LAYER}.mlp.down_proj.weight"
    assert dict(backend.model.named_parameters())[name] is weight
    assert tuple(weight.shape) == (WIDTH, MLP_WIDTH)


def test_find_down_proj_fails_clearly(backend):
    with pytest.raises(ValueError, match="out of range"):
        find_down_proj(backend.model, N_LAYERS)
    with pytest.raises(ValueError, match="decoder layer list"):
        find_down_proj(torch.nn.Linear(4, 4), 0)


def test_find_down_proj_ignores_an_extra_head_with_its_own_layers(backend):
    """Qwen3.5 has a second, one-layer list (`mtp.layers`) that also has mlp.down_proj."""
    model = backend.model
    model.mtp = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([transformers.models.llama.modeling_llama.LlamaDecoderLayer(model.config, 0)])
    name, _ = find_down_proj(model, LAYER)
    assert name == f"model.layers.{LAYER}.mlp.down_proj.weight"


# ---------------------------------------------------------------- apply / remove

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_apply_then_remove_restores_exactly(tokenizer_dir, dtype):
    model = make_model(50, dtype)
    edit = some_edit(model)
    _, weight = find_down_proj(model, LAYER)
    original = weight.detach().clone()

    apply_edit(model, edit)
    expected = (original.float() + edit.alpha * torch.outer(edit.u, edit.v)).to(dtype)
    assert torch.equal(weight, expected)
    assert not torch.equal(weight, original)
    assert weight.dtype == dtype
    change = applied_change(model, edit)
    assert change["weight_changed"] and change["realized_cosine"] > 0.9
    assert change["realized_fraction"] == pytest.approx(1.0, abs=0.1)

    remove_edit(model, edit)
    assert torch.equal(weight, original)


def test_only_one_edit_per_matrix_and_remove_needs_an_applied_edit(backend):
    edit = some_edit(backend.model)
    with pytest.raises(RuntimeError, match="no edit is applied"):
        remove_edit(backend.model, edit)
    apply_edit(backend.model, edit)
    with pytest.raises(RuntimeError, match="already applied"):
        apply_edit(backend.model, edit)
    remove_edit(backend.model, edit)


def test_edited_restores_on_exception(backend):
    edit = some_edit(backend.model)
    _, weight = find_down_proj(backend.model, LAYER)
    original = weight.detach().clone()
    with pytest.raises(KeyError):
        with edited(backend.model, edit):
            assert not torch.equal(weight, original)
            raise KeyError("boom")
    assert torch.equal(weight, original)
    with edited(backend.model, edit):   # and it can be applied again afterwards
        pass
    assert torch.equal(weight, original)


def test_an_edit_that_rounds_away_in_bfloat16_is_reported():
    model = make_model(50, torch.bfloat16)
    edit = some_edit(model, alpha=1e-7)
    with pytest.warns(RuntimeWarning, match="rounded away"):
        apply_edit(model, edit)
    change = applied_change(model, edit)
    assert change == {"weight_changed": False, "realized_fraction": 0.0, "realized_cosine": 0.0, "entries_changed": 0.0}
    remove_edit(model, edit)


# ---------------------------------------------------------------- random control

def test_random_edit_like_has_same_size_and_place(backend):
    edit = some_edit(backend.model)
    control = random_edit_like(edit, seed=7)
    assert (control.alpha, control.alpha_fraction, control.layer, control.param_name) == \
           (edit.alpha, edit.alpha_fraction, edit.layer, edit.param_name)
    assert control.kind == "random" and control.seed == 7 and control.question_id == edit.question_id
    assert control.u.shape == edit.u.shape and control.v.shape == edit.v.shape
    assert control.u.dtype == torch.float32 and control.u.device.type == "cpu"
    assert float(control.u.norm()) == pytest.approx(1.0, abs=1e-5)
    assert float(control.v.norm()) == pytest.approx(1.0, abs=1e-5)
    assert torch.equal(control.u, random_edit_like(edit, seed=7).u)        # repeatable
    assert not torch.equal(control.u, random_edit_like(edit, seed=8).u)    # and depends on the seed
    with edited(backend.model, control):
        pass


# ---------------------------------------------------------------- the targeted edit

def test_gradient_and_input_match_a_direct_computation(backend):
    """x and g from the hook equal what plain autograd gives for the whole model."""
    question = wrong_question(backend, rank=1)
    wrong = wrong_question(backend, rank=0).answer
    prompt = render_prompt(build_prompt(question), str(backend.tokenizer_path))
    name, weight = find_down_proj(backend.model, LAYER)
    x, g, out_norm = edit_module._input_and_margin_gradient(backend, prompt, name, question.answer, wrong)

    seen = {}

    def keep(_module, inputs, output):
        output.retain_grad()
        seen["x"], seen["out"] = inputs[0][0, -1].detach(), output

    module = backend.model.get_submodule(name.removesuffix(".weight"))
    handle = module.register_forward_hook(keep)
    ids = backend.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    logits = backend.model(input_ids=ids).logits[0, -1]
    handle.remove()
    margin = logits[backend._letter_ids[question.answer][0]] - logits[backend._letter_ids[wrong][0]]
    margin.backward()
    backend.model.zero_grad()

    assert torch.allclose(x, seen["x"], atol=1e-6)
    assert torch.allclose(g, seen["out"].grad[0, -1], atol=1e-6, rtol=1e-4)
    assert out_norm == pytest.approx(float(seen["out"][0, -1].detach().norm()), rel=1e-5)
    assert all(p.requires_grad for p in backend.model.parameters())   # flags were put back


def test_compute_targeted_edit_fixes_or_reports(backend):
    question = wrong_question(backend, rank=1)
    name, weight = find_down_proj(backend.model, LAYER)
    original = weight.detach().clone()

    search = compute_targeted_edit(backend, question, LAYER)

    assert torch.equal(weight, original)                 # the model is left unedited
    assert search.question_id == question.id
    assert search.baseline["chosen_letter"] != question.answer and search.baseline["margin"] < 0
    assert search.baseline["reference_norm"] == pytest.approx(spectral_norm(weight))
    assert search.baseline["push_ratio_per_fraction"] >= 1.0 - 1e-4   # ‖ΔW x‖/‖W x‖ >= fraction
    assert [step["alpha_fraction"] for step in search.tried] == list(LADDER)[: len(search.tried)]
    for step in search.tried:
        assert {"alpha_fraction", "chosen_letter", "margin", "prob_correct", "weight_changed"} <= set(step)
    json.dumps(search.tried), json.dumps(search.baseline)   # plain numbers, ready for a results file

    if search.fixed:
        assert search.status == "fixed"
        edit = search.edit
        assert edit.kind == "targeted" and edit.question_id == question.id and edit.param_name == name
        assert edit.alpha == pytest.approx(edit.alpha_fraction * search.baseline["reference_norm"])
        assert edit.alpha_fraction == search.tried[-1]["alpha_fraction"]
        assert float(edit.u.norm()) == pytest.approx(1.0, abs=1e-5)
        assert float(edit.v.norm()) == pytest.approx(1.0, abs=1e-5)
        assert tuple(edit.u.shape) == (WIDTH,) and tuple(edit.v.shape) == (MLP_WIDTH,)
        assert search.tried[-1]["chosen_letter"] == question.answer
        assert all(step["chosen_letter"] != question.answer for step in search.tried[:-1])   # smallest step that works
        prompt = render_prompt(build_prompt(question), str(backend.tokenizer_path))
        with edited(backend.model, edit):
            probs = backend.letter_probabilities(prompt, sorted(question.options), question)
        assert max(probs, key=probs.get) == question.answer
    else:
        assert search.status == "could_not_fix" and search.edit is None
        assert len(search.tried) == len(LADDER)


def test_a_ladder_that_is_too_small_reports_could_not_fix(backend):
    question = wrong_question(backend, rank=-1)   # the letter the model likes least
    search = compute_targeted_edit(backend, question, LAYER, ladder=[1e-6, 2e-6])
    assert not search.fixed and search.edit is None and search.status == "could_not_fix"
    assert [step["alpha_fraction"] for step in search.tried] == [1e-6, 2e-6]
    assert all(not step["is_correct"] for step in search.tried)


def test_an_already_correct_question_is_not_edited(backend):
    question = wrong_question(backend, rank=0)
    search = compute_targeted_edit(backend, question, LAYER)
    assert search.status == "already_correct" and not search.fixed and search.edit is None and search.tried == []


def test_margin_increases_with_alpha(backend, monkeypatch):
    question = wrong_question(backend, rank=-1)
    fractions = [0.02, 0.05, 0.1, 0.2, 0.4]
    # Keep going after the question is fixed, so every step is recorded.
    monkeypatch.setattr(edit_module, "score_answer", _never_correct(edit_module.score_answer))
    search = compute_targeted_edit(backend, question, LAYER, ladder=fractions)
    margins = [search.baseline["margin"]] + [step["margin"] for step in search.tried]
    assert len(margins) == len(fractions) + 1
    assert all(later > earlier for earlier, later in zip(margins, margins[1:]))
    assert all(math.isfinite(margin) for margin in margins)


def _never_correct(score_answer):
    def scored(question, raw_probs):
        answer = score_answer(question, raw_probs)
        answer.is_correct = False
        return answer
    return scored


# ---------------------------------------------------------------- saving

def test_save_and_load_edit_round_trip(backend, tmp_path):
    edit = random_edit_like(some_edit(backend.model), seed=3)
    save_edit(edit, tmp_path / "edits" / "case01")
    assert (tmp_path / "edits" / "case01.json").exists() and (tmp_path / "edits" / "case01.safetensors").exists()
    loaded = load_edit(tmp_path / "edits" / "case01")
    assert torch.equal(loaded.u, edit.u) and torch.equal(loaded.v, edit.v)
    for name in ("layer", "param_name", "alpha", "alpha_fraction", "kind", "question_id", "seed"):
        assert getattr(loaded, name) == getattr(edit, name)


@pytest.mark.parametrize("sharded", [False, True])
def test_saved_model_loads_and_matches_the_in_memory_edit(tokenizer_dir, tmp_path, sharded):
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = make_model(len(tokenizer), torch.bfloat16)
    src = tmp_path / "src"
    model.save_pretrained(str(src), **({"max_shard_size": "20KB"} if sharded else {}))
    tokenizer.save_pretrained(str(src))
    assert (len(list(src.glob("*.safetensors"))) > 1) == sharded
    edit = some_edit(model)

    out = save_edited_model(src, edit, tmp_path / "edited")

    assert out == tmp_path / "edited" and not (tmp_path / "edited.partial").exists()
    rewritten = [p for p in out.glob("*.safetensors") if not p.is_symlink()]
    linked = [p for p in out.glob("*.safetensors") if p.is_symlink()]
    assert len(rewritten) == 1 and len(linked) == len(list(src.glob("*.safetensors"))) - 1
    assert json.loads((out / "selma_edit.json").read_text())["weight_changed"] is True
    if sharded:
        assert (out / "model.safetensors.index.json").read_text() == (src / "model.safetensors.index.json").read_text()
    for small in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        assert (out / small).is_file() and not (out / small).is_symlink()

    loaded = transformers.AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    transformers.AutoTokenizer.from_pretrained(str(out))
    with edited(model, edit):
        in_memory = {name: p.detach().clone() for name, p in model.named_parameters()}
    on_disk = dict(loaded.named_parameters())
    assert on_disk.keys() == in_memory.keys()
    for name, tensor in in_memory.items():     # bit for bit, the edited matrix and everything else
        assert on_disk[name].dtype == tensor.dtype and torch.equal(on_disk[name], tensor), name
    original = dict(model.named_parameters())[edit.param_name]
    assert not torch.equal(on_disk[edit.param_name], original)

    with pytest.raises(FileExistsError):
        save_edited_model(src, edit, tmp_path / "edited")


def test_saved_model_finds_a_weight_stored_under_a_reordered_name(tmp_path):
    """Gemma 3 files say `language_model.model.layers...`; the loaded model says `model.language_model.layers...`."""
    from safetensors.torch import load_file, save_file

    src = tmp_path / "src"
    src.mkdir()
    stored = "language_model.model.layers.2.mlp.down_proj.weight"
    weight = torch.randn(WIDTH, MLP_WIDTH).to(torch.bfloat16)
    save_file({stored: weight, "language_model.model.layers.1.mlp.down_proj.weight": weight.clone()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    edit = Edit(layer=2, param_name="model.language_model.layers.2.mlp.down_proj.weight",
                u=torch.eye(WIDTH)[0], v=torch.eye(MLP_WIDTH)[0], alpha=1.0, alpha_fraction=0.1,
                kind="targeted", question_id=None, seed=None)
    out = save_edited_model(src, edit, tmp_path / "out")
    saved = load_file(str(out / "model.safetensors"))
    assert not torch.equal(saved[stored], weight)
    assert torch.equal(saved["language_model.model.layers.1.mlp.down_proj.weight"], weight)

    missing = Edit(**{**edit_module.asdict_shallow(edit), "param_name": "model.language_model.layers.9.mlp.down_proj.weight"})
    with pytest.raises(ValueError, match="cannot find"):
        save_edited_model(src, missing, tmp_path / "out2")


def test_saving_an_edit_that_rounds_away_warns(tmp_path):
    model = make_model(50, torch.bfloat16)
    model.save_pretrained(str(tmp_path / "src"))
    with pytest.warns(RuntimeWarning, match="rounded away"):
        out = save_edited_model(tmp_path / "src", some_edit(model, alpha=1e-7), tmp_path / "out")
    assert json.loads((out / "selma_edit.json").read_text())["weight_changed"] is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        save_edited_model(tmp_path / "src", some_edit(model, alpha=0.5), tmp_path / "out_ok")
