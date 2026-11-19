"""Tests for the editor workbench, on a tiny random model in a fake project folder (CPU, offline).

Skipped when torch is not installed. Every forward pass starts a worker process, as on the GPU
machine, so the tests that run passes take a few seconds each.

Run with:  uv run --extra gpu pytest tests/test_editor_workbench.py
"""

import json
import random
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
pytest.importorskip("accelerate")

from safetensors.torch import load_file  # noqa: E402

from bench.backends.hf import _letter_token_ids, _mass  # noqa: E402
from bench.editor import (  # noqa: E402
    AlreadySubmitted,
    BudgetExhausted,
    Layout,
    NotPracticeQuestion,
    Workbench,
    WorkbenchError,
    load_edit,
    permute_options,
    save_edited_model,
)
from bench.editor.workbench import applied_edit  # noqa: E402
from bench.prompts import build_prompt, render_prompt  # noqa: E402
from bench.records import Question, write_jsonl  # noqa: E402
from bench.run import score_answer  # noqa: E402

WIDTH, MLP_WIDTH, N_LAYERS = 32, 64, 4
WORDS = ["olik", "liknande", "snabb", "sen", "lugn", "glad", "trött", "stor", "liten", "varm", "kall", "ny"]
DOWN_PROJ = "model.layers.2.mlp.down_proj.weight"


def make_question(date: str, number: int) -> Question:
    rng = random.Random(f"{date}/{number}")
    word, *options = rng.sample(WORDS, 6)
    return Question(id=f"{date}/p1/ORD/{number}", exam_date=date, exam_pass=1, section="ORD", number=number,
                    question=word, options=dict(zip("ABCDE", options)), answer=rng.choice("ABCDE"))


# One exam date per part of the split; "test" questions are in questions.jsonl but in no group of split.json.
PAPERS = {"drill": "2020-01-01", "validation": "2021-01-01", "hidden": "2022-01-01", "test": "2023-01-01"}
QUESTIONS = {paper: [make_question(date, number) for number in range(1, 7)] for paper, date in PAPERS.items()}
PRACTICE = QUESTIONS["drill"] + QUESTIONS["validation"]


def load_model(model_dir):
    return transformers.LlamaForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).eval()


def next_token_probs(model, tokenizer, text):
    input_ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.inference_mode():
        return torch.softmax(model(input_ids=input_ids, logits_to_keep=1).logits[0, -1].float(), dim=-1)


def bench_answer(model, tokenizer, tokenizer_dir, question):
    """The bench's own scoring of one question (the rule of bench.backends.hf), in this process."""
    probs = next_token_probs(model, tokenizer, render_prompt(build_prompt(question), str(tokenizer_dir))).numpy()
    letter_ids = _letter_token_ids(tokenizer, tuple("ABCDE"))
    return score_answer(question, {letter: _mass(probs, letter_ids[letter]) for letter in sorted(question.options)})


@pytest.fixture(scope="module")
def layout(tmp_path_factory):
    """A fake project: tiny bfloat16 model folder, questions, split and the unedited model's answers."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    root = tmp_path_factory.mktemp("project")
    model_dir = root / "models" / "tiny"

    body = build_prompt(PRACTICE[0])[0]["content"]
    words = [word for word, _ in pre_tokenizers.Whitespace().pre_tokenize_str(body + " user assistant :")]
    vocab = {"[UNK]": 0}
    for word in [*words, *WORDS, "A", "B", "C", "D", "E"]:
        vocab.setdefault(word, len(vocab))
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}user : {{ m['content'] }} {% endfor %}assistant :"
    tokenizer.save_pretrained(str(model_dir))

    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=WIDTH, intermediate_size=MLP_WIDTH, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        initializer_range=0.1,   # larger than the default, so the answer depends on the question
    )
    transformers.LlamaForCausalLM(config).to(torch.bfloat16).save_pretrained(str(model_dir))

    write_jsonl(root / "questions.jsonl", [question for group in QUESTIONS.values() for question in group])
    model = load_model(model_dir)
    question_ids = {}
    for paper in ("drill", "validation", "hidden"):
        answers = [bench_answer(model, tokenizer, model_dir, question) for question in QUESTIONS[paper]]
        write_jsonl(root / "baseline" / paper / "answers.jsonl", answers)
        question_ids[f"{paper}_wrong"] = [answer.question_id for answer in answers if not answer.is_correct]
        question_ids[f"{paper}_right"] = [answer.question_id for answer in answers if answer.is_correct]
    (root / "split.json").write_text(json.dumps({"question_ids": question_ids}), encoding="utf-8")

    return Layout(model="tiny", model_dir=model_dir, tokenizer_dir=model_dir, dtype="bfloat16",
                  questions=root / "questions.jsonl", split=root / "split.json", baseline=root / "baseline",
                  runs_dir=root / "editor_runs")


@pytest.fixture(autouse=True)
def workers_on_cpu(monkeypatch):
    """Worker processes inherit this, so on a GPU machine they still compute on the CPU like this process."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


@pytest.fixture()
def run_name(request):
    return re_safe(request.node.name)


def re_safe(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name)


def column_edit(seed=0, scale=0.5, cols=(3, 10, 40)):
    values = torch.randn(WIDTH, len(cols), generator=torch.Generator().manual_seed(seed)) * scale
    return {DOWN_PROJ: {"cols": list(cols), "values": values}}


# ---------------------------------------------------------------- free tools

def test_weights_are_the_models_parameters_and_read_only(layout, run_name):
    wb = Workbench.create(run_name, budget=1, layout=layout)
    weights = wb.weights()
    model = load_model(layout.model_dir)
    parameters = dict(model.named_parameters(remove_duplicate=False))

    assert set(weights.names()) <= set(parameters)
    assert {name for name in parameters if name not in weights} <= {"lm_head.weight", "model.embed_tokens.weight"}
    for name in weights:
        assert weights.dtype(name) == "bfloat16"
        assert torch.equal(weights[name], parameters[name])
    assert weights.total_weights() == sum(p.numel() for p in model.parameters())
    assert weights.float32(DOWN_PROJ).dtype == torch.float32
    assert wb.layer_param(2) == DOWN_PROJ
    assert wb.config["hidden_size"] == WIDTH

    weights[DOWN_PROJ].zero_()   # changes a copy only
    assert torch.equal(weights[DOWN_PROJ], parameters[DOWN_PROJ])
    with pytest.raises(KeyError):
        weights["model.layers.99.mlp.down_proj.weight"]

    ids = wb.token_ids("glad kall")
    assert len(ids) == 2
    assert torch.equal(wb.embedding_rows("glad kall"), parameters["model.embed_tokens.weight"][ids].float())
    assert torch.equal(wb.unembedding_rows(ids), parameters["lm_head.weight"][ids].float())
    assert wb.passes_used == 0   # nothing above ran the model


def test_practice_data_excludes_every_other_paper(layout, run_name):
    wb = Workbench.create(run_name, budget=1, layout=layout)
    practice = wb.practice_questions()
    assert [item.id for item in practice] == [question.id for question in PRACTICE]
    assert {item.paper for item in practice} == {"drill", "validation"}
    assert all(item.baseline.question_id == item.id and item.baseline.correct_letter == item.question.answer
               for item in practice)

    question = PRACTICE[0]
    assert wb.render_prompt(question) == render_prompt(build_prompt(question), str(layout.tokenizer_dir))
    shown = permute_options(question, "EDCBA")
    assert wb.render_prompt(question.id, "EDCBA") == render_prompt(build_prompt(shown), str(layout.tokenizer_dir))
    assert shown.options["A"] == question.options["E"]
    assert shown.options[shown.answer] == question.options[question.answer]
    with pytest.raises(WorkbenchError):
        permute_options(question, "AABCD")

    for outside in (QUESTIONS["hidden"][0], QUESTIONS["test"][0]):
        with pytest.raises(NotPracticeQuestion):
            wb.render_prompt(outside)
        with pytest.raises(NotPracticeQuestion):
            wb.answer([PRACTICE[0], outside])
    assert wb.passes_used == 0 and wb.ledger() == []   # a refused call costs nothing


def test_create_refuses_an_existing_run_and_bad_arguments(layout, run_name):
    Workbench.create(run_name, budget=5, layout=layout)
    with pytest.raises(WorkbenchError):
        Workbench.create(run_name, budget=50, layout=layout)
    assert Workbench.open(run_name, layout.runs_dir).budget == 5
    with pytest.raises(WorkbenchError):
        Workbench.create("../escape", budget=5, layout=layout)
    with pytest.raises(WorkbenchError):
        Workbench.create(run_name + "_zero", budget=0, layout=layout)
    with pytest.raises(WorkbenchError):
        Workbench.open("no_such_run", layout.runs_dir)


# ---------------------------------------------------------------- edits

def test_edit_formats_are_checked(layout, run_name):
    wb = Workbench.create(run_name, budget=1, layout=layout)
    dense = wb.make_edit({DOWN_PROJ: torch.ones(WIDTH, MLP_WIDTH, dtype=torch.float64)})
    assert dense[DOWN_PROJ].dtype == torch.float32
    rows = wb.make_edit({DOWN_PROJ: {"rows": [0, 5], "values": torch.ones(2, MLP_WIDTH)}})
    assert rows[DOWN_PROJ]["rows"].tolist() == [0, 5]
    norm = wb.make_edit({"model.norm.weight": {"rows": [1], "values": [0.5]}})   # rows also index a vector
    assert norm["model.norm.weight"]["values"].shape == (1,)

    bad_edits = [
        {},
        {"model.layers.9.mlp.down_proj.weight": torch.ones(WIDTH, MLP_WIDTH)},      # unknown name
        {DOWN_PROJ: torch.ones(MLP_WIDTH, WIDTH)},                                  # wrong shape
        {DOWN_PROJ: {"cols": [1, 2], "values": torch.ones(2, WIDTH)}},              # values the wrong way round
        {DOWN_PROJ: {"cols": [1, 1], "values": torch.ones(WIDTH, 2)}},              # index twice
        {DOWN_PROJ: {"cols": [MLP_WIDTH], "values": torch.ones(WIDTH, 1)}},         # out of range
        {DOWN_PROJ: {"rows": [0], "cols": [1], "values": torch.ones(1, 1)}},        # both axes
        {DOWN_PROJ: {"cols": [1], "values": torch.full((WIDTH, 1), float("nan"))}},
        {"model.norm.weight": {"cols": [1], "values": torch.ones(1, 1)}},           # cols on a vector
    ]
    for bad in bad_edits:
        with pytest.raises(WorkbenchError):
            wb.make_edit(bad)


def test_edit_size_counts_touched_and_really_changed_weights(layout, run_name):
    wb = Workbench.create(run_name, budget=1, layout=layout)
    weight = wb.weights().float32(DOWN_PROJ)
    values = torch.zeros(WIDTH, 2)
    values[:, 0] = 1.0                        # large: every entry of column 3 changes
    values[:8, 1] = weight[:8, 10] * 1e-4     # far below bfloat16's resolution: rounded away
    size = wb.edit_size({DOWN_PROJ: {"cols": [3, 10], "values": values}})

    info = size["tensors"][DOWN_PROJ]
    assert info["form"] == "cols"
    assert info["weights_touched"] == size["weights_touched"] == WIDTH + int((values[:, 1] != 0).sum())
    assert info["weights_changed"] == size["weights_changed"] == WIDTH
    assert size["model_weights"] == wb.weights().total_weights()
    assert size["share_of_model"] == pytest.approx(size["weights_touched"] / size["model_weights"])
    assert info["delta_norm"] == pytest.approx(float(values.norm()))
    assert info["relative_norm"] == pytest.approx(float(values.norm() / weight.norm()))


@pytest.mark.parametrize("form", ["dense", "rows", "cols"])
def test_an_applied_edit_is_exact_and_fully_restored(layout, tmp_path, form):
    generator = torch.Generator().manual_seed(1)
    change = {
        "dense": torch.randn(WIDTH, MLP_WIDTH, generator=generator) * 0.1,
        "rows": {"rows": [7, 2], "values": torch.randn(2, MLP_WIDTH, generator=generator) * 0.1},
        "cols": {"cols": [40, 3, 10], "values": torch.randn(WIDTH, 3, generator=generator) * 0.1},
    }[form]
    edit = {DOWN_PROJ: change, "model.norm.weight": {"rows": [4], "values": torch.tensor([0.25])}}

    model = load_model(layout.model_dir)
    before = {name: parameter.clone() for name, parameter in model.named_parameters()}
    full_delta = torch.zeros(WIDTH, MLP_WIDTH)
    if form == "dense":
        full_delta += change
    elif form == "rows":
        full_delta[change["rows"]] = change["values"]
    else:
        full_delta[:, change["cols"]] = change["values"]
    expected = (before[DOWN_PROJ].float() + full_delta).to(torch.bfloat16)

    with torch.inference_mode(), applied_edit(model, edit):
        during = {name: parameter.clone() for name, parameter in model.named_parameters()}
    after = dict(model.named_parameters())

    assert torch.equal(during[DOWN_PROJ], expected) and not torch.equal(expected, before[DOWN_PROJ])
    assert during["model.norm.weight"][4] == before["model.norm.weight"][4] + 0.25
    assert [name for name in before if not torch.equal(during[name], before[name])] == sorted(edit)
    assert all(torch.equal(after[name], before[name]) for name in before)   # bit for bit
    assert not any(parameter.requires_grad and parameter.grad is not None for parameter in after.values())

    # The model folder written for the evaluation holds exactly the numbers of the edited model in memory.
    out_dir = save_edited_model(layout.model_dir, edit, tmp_path / "edited")
    saved = dict(load_model(out_dir).named_parameters())
    assert all(torch.equal(saved[name], during[name]) for name in before)


# ---------------------------------------------------------------- forward passes

def test_answer_scores_like_the_bench_and_writes_the_ledger(layout, run_name):
    wb = Workbench.create(run_name, budget=2, layout=layout)
    practice = wb.practice_questions()
    reversed_orders = ["".join(reversed(sorted(item.question.options))) for item in practice]

    # One pass: every question in the stored order, then again with the options reversed.
    answers = wb.answer(practice + practice, options_orders=[None] * len(practice) + reversed_orders)
    plain, permuted = answers[:len(practice)], answers[len(practice):]

    for item, answer in zip(practice, plain):
        assert answer.question_id == item.id and answer.paper == item.paper and answer.options_order is None
        assert answer.letter_probs == item.baseline.letter_probs          # the unedited model, exactly
        assert (answer.chosen_letter, answer.is_correct) == (item.baseline.chosen_letter, item.baseline.is_correct)
        assert answer.chosen_original == answer.chosen_letter

    model, tokenizer = load_model(layout.model_dir), wb.tokenizer
    for item, order, answer in zip(practice, reversed_orders, permuted):
        shown = permute_options(item.question, order)
        expected = bench_answer(model, tokenizer, layout.tokenizer_dir, shown)
        assert answer.options_order == order
        assert answer.letter_probs == expected.letter_probs
        assert answer.correct_letter == shown.answer and answer.is_correct == expected.is_correct
        assert shown.options[answer.chosen_letter] == item.question.options[answer.chosen_original]

    (entry,) = wb.ledger()
    assert entry["kind"] == "answer" and entry["cost"] == 1 and entry["n_items"] == 2 * len(practice)
    assert entry["edit"] is None and entry["passes_used"] == 1 and entry["passes_left"] == 1
    assert "time" in entry
    assert (wb.passes_used, wb.passes_left) == (1, 1)
    assert wb.load_pass(1) == answers   # an earlier result can be read back for free

    with pytest.raises(WorkbenchError):
        wb.answer([])
    with pytest.raises(WorkbenchError):
        wb.answer(practice[:2], options_orders=["ABCDE"])
    assert wb.passes_used == 1


def test_read_with_an_edit_matches_the_edited_model(layout, run_name):
    wb = Workbench.create(run_name, budget=1, layout=layout)
    edit = column_edit()
    texts = [wb.render_prompt(PRACTICE[0]), "glad kall stor", "liten"]
    result = wb.read(texts, edit=edit, layers=[1, 2], top_k=5)

    assert result.texts == texts and result.layers == [1, 2]
    assert result.n_tokens[1:] == [3, 1]
    assert result.top_ids.shape == result.top_probs.shape == (3, 5)
    assert result.mlp_acts.shape == (3, 2, MLP_WIDTH) and result.residual.shape == (3, 2, WIDTH)
    assert result.top_tokens[0] == [wb.tokenizer.decode([token]) for token in result.top_ids[0].tolist()]

    # The same quantities from the model edited in this process.
    model, captured = load_model(layout.model_dir), {}
    layer = model.model.layers[2]
    layer.mlp.down_proj.register_forward_pre_hook(lambda module, args: captured.update(mlp=args[0][0, -1].float()))
    layer.register_forward_hook(lambda module, args, output: captured.update(
        residual=(output[0] if isinstance(output, tuple) else output)[0, -1].float()))
    unedited_probs = next_token_probs(model, wb.tokenizer, texts[1])
    with applied_edit(model, edit):
        probs = next_token_probs(model, wb.tokenizer, texts[1])
    top = torch.topk(probs, 5)
    assert torch.equal(result.top_ids[1], top.indices) and torch.equal(result.top_probs[1], top.values)
    assert torch.equal(result.mlp_acts[1, 1], captured["mlp"]) and torch.equal(result.residual[1, 1], captured["residual"])
    assert not torch.equal(probs, unedited_probs)   # the edit was really in the pass

    (entry,) = wb.ledger()
    assert entry["kind"] == "read" and entry["n_items"] == 3
    assert entry["edit"][DOWN_PROJ]["form"] == "cols" and entry["edit"][DOWN_PROJ]["weights_touched"] == 3 * WIDTH

    again = wb.load_pass(1)
    assert torch.equal(again.mlp_acts, result.mlp_acts) and again.top_tokens == result.top_tokens
    for bad_call in (lambda: wb.read([]), lambda: wb.read(["glad"], layers=[N_LAYERS]), lambda: wb.read([""]),
                     lambda: wb.read(["glad"], top_k=0)):
        with pytest.raises(WorkbenchError):
            bad_call()


def test_budget_holds_across_instances_and_processes(layout, run_name):
    wb = Workbench.create(run_name, budget=2, layout=layout)
    wb.answer(PRACTICE[:2])

    # A second pass from a brand-new Python process: it continues the same budget.
    script = ("import sys; from bench.editor import Workbench; "
              "wb = Workbench.open(sys.argv[1], sys.argv[2]); wb.read(['glad kall']); print(wb.passes_left)")
    other = subprocess.run([sys.executable, "-c", script, run_name, str(layout.runs_dir)],
                           capture_output=True, text=True, check=True)
    assert other.stdout.strip().splitlines()[-1] == "0"

    for bench in (wb, Workbench.open(run_name, layout.runs_dir)):
        assert bench.passes_left == 0
        with pytest.raises(BudgetExhausted):
            bench.answer(PRACTICE[:1])
        with pytest.raises(BudgetExhausted):
            bench.read(["glad"])
    refused = subprocess.run([sys.executable, "-c", script, run_name, str(layout.runs_dir)], capture_output=True, text=True)
    assert refused.returncode != 0 and "BudgetExhausted" in refused.stderr

    assert [entry["kind"] for entry in wb.ledger()] == ["answer", "read"]
    assert [entry["passes_left"] for entry in wb.ledger()] == [1, 0]


def test_a_failed_pass_is_not_charged(layout, run_name, tmp_path):
    broken = Layout(**{**layout.__dict__, "dtype": "no_such_dtype", "runs_dir": tmp_path})
    wb = Workbench.create(run_name, budget=1, layout=broken)
    with pytest.raises(WorkbenchError, match="not charged"):
        wb.read(["glad"])
    (entry,) = wb.ledger()
    assert entry["kind"] == "failed" and entry["cost"] == 0 and wb.passes_left == 1


# ---------------------------------------------------------------- handing in

def test_submit_works_once_and_closes_the_run(layout, run_name):
    wb = Workbench.create(run_name, budget=3, layout=layout)
    edit = column_edit()
    with pytest.raises(WorkbenchError):
        wb.submit(edit, explanation="  ")

    record = wb.submit(edit, explanation="Subtracts a direction from three MLP output columns.")
    assert record["passes_used"] == 0 and record["budget"] == 3
    assert record["size"]["weights_touched"] == 3 * WIDTH

    saved = json.loads((wb.run_dir / "final_edit.json").read_text(encoding="utf-8"))
    assert saved["explanation"].startswith("Subtracts") and saved["size"]["tensors"][DOWN_PROJ]["form"] == "cols"
    loaded = load_edit(wb.run_dir / "final_edit.safetensors")
    assert loaded[DOWN_PROJ]["cols"].tolist() == [3, 10, 40]
    assert torch.equal(loaded[DOWN_PROJ]["values"], edit[DOWN_PROJ]["values"])
    assert set(load_file(str(wb.run_dir / "final_edit.safetensors"))) == {f"values:{DOWN_PROJ}", f"cols:{DOWN_PROJ}"}

    for bench in (wb, Workbench.open(run_name, layout.runs_dir)):
        with pytest.raises(AlreadySubmitted):
            bench.submit(column_edit(seed=5), explanation="A second try.")
        with pytest.raises(AlreadySubmitted):
            bench.read(["glad"])
    assert torch.equal(load_edit(wb.run_dir / "final_edit.safetensors")[DOWN_PROJ]["values"], edit[DOWN_PROJ]["values"])
    assert wb.ledger()[-1]["kind"] == "submit" and wb.passes_used == 0
