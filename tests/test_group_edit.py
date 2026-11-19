"""Tests for the trained group edit, on a tiny random model (CPU, offline). Skipped when torch is not installed.

Run with:  uv run --extra gpu pytest tests/test_group_edit.py
"""

import json
import random

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

from bench.edit import find_down_proj, letter_token_ids  # noqa: E402
from bench.group_edit import (  # noqa: E402
    GroupDelta,
    applied,
    default_group_layers,
    delta_norms,
    letter_cross_entropy,
    load_delta,
    save_snapshots,
    random_delta_like,
    save_delta,
    save_merged_model,
    shuffle_options,
    train_group_delta,
)
from bench.prompts import build_prompt, render_prompt  # noqa: E402
from bench.records import Question  # noqa: E402
from bench.run import score_question  # noqa: E402

LAYERS = [1, 2]
RANK = 4
WIDTH, MLP_WIDTH, N_LAYERS = 32, 64, 4
# Larger random weights than the default 0.02, so that the tiny model's answer depends on the
# question it is shown (with the default, attention is nearly uniform and it answers every prompt alike).
INITIALIZER_RANGE = 0.1
RULE_WORDS = ["hund", "katt"]
WORDS = ["olik", "liknande", "snabb", "sen", "lugn", "glad", "trött", "stor", "liten", "varm", "kall", "ny"]


def make_question(number: int, rng: random.Random) -> Question:
    word, *options = rng.sample(WORDS, 6)
    return Question(
        id=f"2025-10-19/p3/ORD/{number}", exam_date="2025-10-19", exam_pass=3, section="ORD", number=number,
        question=word, options=dict(zip("ABCDE", options)), answer="A",
    )


QUESTIONS = [make_question(number, random.Random(number)) for number in range(1, 13)]


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
    """A word-level tokenizer that knows the words of the test prompts, with a chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    body = build_prompt(QUESTIONS[0])[0]["content"] + " " + build_prompt(make_rule_question(0, "hund", "A"))[0]["content"]
    words = [word for word, _ in pre_tokenizers.Whitespace().pre_tokenize_str(body + " user assistant :")]
    vocab = {"[UNK]": 0}
    for word in [*words, *WORDS, *RULE_WORDS, "A", "B", "C", "D", "E"]:
        vocab.setdefault(word, len(vocab))
    raw = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}user : {{ m['content'] }} {% endfor %}assistant :"
    path = tmp_path_factory.mktemp("tokenizer")
    tokenizer.save_pretrained(str(path))
    return path


def make_model(vocab_size, dtype=torch.float32, seed=0, initializer_range=INITIALIZER_RANGE):
    torch.manual_seed(seed)
    config = transformers.LlamaConfig(
        vocab_size=vocab_size, hidden_size=WIDTH, intermediate_size=MLP_WIDTH, num_hidden_layers=N_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        initializer_range=initializer_range,
    )
    return transformers.LlamaForCausalLM(config).to(dtype).eval()


@pytest.fixture()
def backend(tokenizer_dir):
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))
    return TinyBackend(make_model(len(tokenizer)), tokenizer, tokenizer_dir)


def rule_task(backend):
    """A task the tiny model can learn and that carries over to unseen questions.

    Every question asks about one of two words. The key follows a rule: for the first word it
    is the letter the unedited model prefers, for the second word another letter. So the model
    starts out right on the first kind and wrong on the second, and "the second word means the
    other letter" is something a small change can teach it. Returns (wrong, right, validation);
    validation holds unseen questions of both kinds (MEK) and two ORD questions.
    """
    probe = score_question(backend, make_rule_question(0, RULE_WORDS[0], "A"), str(backend.tokenizer_path))
    ranked = sorted(probe.letter_probs, key=probe.letter_probs.get, reverse=True)
    keys = {RULE_WORDS[0]: ranked[0], RULE_WORDS[1]: ranked[2]}
    questions = [make_rule_question(n, RULE_WORDS[n % 2], keys[RULE_WORDS[n % 2]]) for n in range(1, 41)]
    answers = [score_question(backend, q, str(backend.tokenizer_path)) for q in questions]
    drill, held_out = list(zip(questions, answers))[:24], questions[24:]
    wrong = [q for q, a in drill if not a.is_correct and q.question == RULE_WORDS[1]]
    right = [q for q, a in drill if a.is_correct and q.question == RULE_WORDS[0]]
    assert len(wrong) >= 8 and len(right) >= 8, (len(wrong), len(right))
    ords = [Question(**{**q.__dict__, "id": q.id.replace("MEK", "ORD"), "section": "ORD"}) for q in held_out[:2]]
    return wrong, right, held_out[2:] + ords


def make_rule_question(number: int, word: str, answer: str) -> Question:
    rng = random.Random(number)
    return Question(
        id=f"2025-10-19/p3/MEK/{number}", exam_date="2025-10-19", exam_pass=3, section="MEK", number=number,
        question=word, options=dict(zip("ABCDE", rng.sample(WORDS, 5))), answer=answer,
    )


def some_delta(model, layers=LAYERS, rank=RANK, seed=1, scale=0.05) -> GroupDelta:
    generator = torch.Generator().manual_seed(seed)
    names, A, B = [], [], []
    for layer in layers:
        name, weight = find_down_proj(model, layer)
        names.append(name)
        A.append(torch.randn(weight.shape[0], rank, generator=generator) * scale)
        B.append(torch.randn(rank, weight.shape[1], generator=generator))
    return GroupDelta(layers=list(layers), param_names=names, A=A, B=B, rank=rank, kind="trained", seed=0,
                      info={"layers": [{"layer": layer, "reference_spectral_norm": 2.0} for layer in layers]})


def weights(model, delta):
    return [dict(model.named_parameters())[name] for name in delta.param_names]


# ---------------------------------------------------------------- layers and shuffling

def test_default_group_layers_end_one_above_the_edit_layer():
    assert default_group_layers(34) == [21, 22, 23, 24]   # Gemma 3 4B
    assert default_group_layers(32) == [19, 20, 21, 22]   # Qwen3.5-9B
    assert default_group_layers(34, count=2) == [23, 24]
    assert default_group_layers(4) == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        default_group_layers(4, count=5)


def test_shuffle_options_relabels_the_target():
    question = QUESTIONS[0]
    correct_text = question.options[question.answer]
    rng = random.Random(0)
    answers = set()
    for _ in range(50):
        shuffled = shuffle_options(question, rng)
        assert sorted(shuffled.options) == sorted(question.options)                      # still A..E
        assert sorted(shuffled.options.values()) == sorted(question.options.values())    # same option texts
        assert shuffled.options[shuffled.answer] == correct_text                         # the key follows the option
        assert (shuffled.id, shuffled.question) == (question.id, question.question)
        answers.add(shuffled.answer)
    assert answers == set("ABCDE")                         # every position is used
    assert question.answer == "A" and question.options["A"] == correct_text   # the original is untouched
    four = Question(**{**question.__dict__, "options": {k: v for k, v in question.options.items() if k != "E"}})
    assert sorted(shuffle_options(four, rng).options) == list("ABCD")


def test_letter_cross_entropy_matches_the_backend_probabilities(backend):
    question = QUESTIONS[0]
    answer = score_question(backend, question, str(backend.tokenizer_path))
    loss = letter_cross_entropy(backend, question)
    assert loss.item() == pytest.approx(-torch.log(torch.tensor(answer.prob_correct)).item(), rel=1e-4)


# ---------------------------------------------------------------- apply / remove

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_applied_merges_and_restores_exactly(dtype):
    model = make_model(50, dtype)
    delta = some_delta(model)
    originals = [w.detach().clone() for w in weights(model, delta)]

    with applied(model, delta) as change:
        for weight, original, A, B in zip(weights(model, delta), originals, delta.A, delta.B):
            assert weight.dtype == dtype and not torch.equal(weight, original)
            assert torch.allclose(weight.float(), (original.float() + A @ B).to(dtype).float(), atol=1e-5)
        assert [entry["layer"] for entry in change] == LAYERS
        assert all(entry["realized_fraction"] == pytest.approx(1.0, abs=0.1) for entry in change)
        assert all(entry["entries_changed"] > 0.5 for entry in change)
        with pytest.raises(RuntimeError, match="already applied"):
            with applied(model, delta):
                pass
    for weight, original in zip(weights(model, delta), originals):
        assert torch.equal(weight, original)


def test_applied_restores_on_exception(backend):
    delta = some_delta(backend.model)
    originals = [w.detach().clone() for w in weights(backend.model, delta)]
    with pytest.raises(KeyError):
        with applied(backend.model, delta):
            raise KeyError("boom")
    for weight, original in zip(weights(backend.model, delta), originals):
        assert torch.equal(weight, original)
    with applied(backend.model, delta):   # and it can be applied again afterwards
        pass

    bad = some_delta(backend.model)
    bad.A[1] = bad.A[1][:-1]              # the second layer does not fit: the first must be put back
    with pytest.raises(ValueError, match="does not fit"):
        with applied(backend.model, bad):
            pass
    for weight, original in zip(weights(backend.model, delta), originals):
        assert torch.equal(weight, original)
    with applied(backend.model, delta):
        pass


def test_a_delta_that_rounds_away_in_bfloat16_is_reported():
    model = make_model(50, torch.bfloat16)
    with applied(model, some_delta(model, scale=1e-9)) as change:
        assert all(entry == {"layer": entry["layer"], "realized_fraction": 0.0, "entries_changed": 0.0}
                   for entry in change)


# ---------------------------------------------------------------- random control

def test_random_delta_like_matches_frobenius_norms(backend):
    delta = some_delta(backend.model)
    control = random_delta_like(delta, seed=7)
    assert (control.layers, control.param_names, control.rank) == (delta.layers, delta.param_names, delta.rank)
    assert control.kind == "random" and control.seed == 7
    for A, B, cA, cB, info in zip(delta.A, delta.B, control.A, control.B, control.info["layers"]):
        assert cA.shape == A.shape and cB.shape == B.shape and cA.dtype == torch.float32
        assert float(torch.linalg.norm(cA @ cB)) == pytest.approx(float(torch.linalg.norm(A @ B)), rel=1e-4)
        assert not torch.allclose(cA @ cB, A @ B)
        assert info["frobenius_norm"] == pytest.approx(float(torch.linalg.norm(A @ B)), rel=1e-4)
        assert info["relative_norm"] == pytest.approx(info["spectral_norm"] / 2.0)
    assert torch.equal(control.A[0], random_delta_like(delta, seed=7).A[0])       # repeatable
    assert not torch.equal(control.A[0], random_delta_like(delta, seed=8).A[0])   # and depends on the seed
    json.dumps(control.info)
    with applied(backend.model, control):
        pass


def test_delta_norms_match_the_full_matrix():
    A, B = torch.randn(32, 4), torch.randn(4, 64)
    frobenius, spectral = delta_norms(A, B)
    assert frobenius == pytest.approx(float(torch.linalg.norm(A @ B)), rel=1e-4)
    assert spectral == pytest.approx(float(torch.linalg.matrix_norm(A @ B, ord=2)), rel=1e-4)
    assert delta_norms(A, torch.zeros(4, 64)) == (0.0, 0.0)


# ---------------------------------------------------------------- training

def train(backend, task, log_path, **overrides):
    """train_group_delta on the rule task, with settings the tiny model learns it under."""
    wrong, right, validation = task
    # shuffle=False: the rule is about letters, which shuffling would (rightly) make unlearnable.
    # right_kept_floor=0: on the way to the rule the tiny model first answers everything alike.
    settings = dict(layers=LAYERS, rank=RANK, lr=3e-3, eval_every=10, max_steps=200, seed=0,
                    validation=validation, shuffle=False, right_kept_floor=0.0)
    return train_group_delta(backend, wrong, right, log_path, **{**settings, **overrides})


def read_log(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_training_improves_validation_and_returns_the_best_checkpoint(backend, tmp_path):
    task = wrong, right, validation = rule_task(backend)
    model = backend.model
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    log_path = tmp_path / "logs" / "train.jsonl"

    result = train(backend, task, log_path)

    rows = read_log(log_path)
    assert rows[0]["step"] == 0 and rows[0]["loss"] is None and rows[0]["is_best"]
    assert rows[0]["drill_wrong_fixed"] == 0.0 and rows[0]["drill_right_kept"] == 1.0
    assert rows[0]["layers"][0]["frobenius_norm"] == 0.0      # training starts from the unedited model
    assert [row["step"] for row in rows[1:]] == list(range(10, result.steps + 1, 10))
    assert rows[-1]["loss_wrong"] < rows[1]["loss_wrong"]
    assert max(row["drill_wrong_fixed"] for row in rows) > 0.5
    assert {"layer", "frobenius_norm", "spectral_norm", "relative_norm", "realized_fraction"} <= set(rows[-1]["layers"][0])
    start = rows[0]["validation"]
    assert start["n"] == len(validation) and start["n_headline"] == sum(q.section == "MEK" for q in validation)
    assert set(start["by_section"]) == {"MEK", "ORD"}

    # The returned delta is the checkpoint with the best validation headline accuracy
    # (ties: higher mean probability of the correct option), and that got better than the start.
    best = max(rows, key=lambda row: (row["validation"]["headline_accuracy"], row["validation"]["mean_prob_correct"], -row["step"]))
    assert result.final["step"] == best["step"] > 0
    assert result.final["validation"] == best["validation"]
    assert best["validation"]["headline_accuracy"] > start["headline_accuracy"]
    assert [row["step"] for row in rows if row["is_best"]][-1] == best["step"]
    assert result.final["n_wrong_fixed"] == best["n_wrong_fixed"] and result.final["n_right_kept"] == best["n_right_kept"]
    assert result.stop_reason in ("no_improvement", "right_kept_low", "max_steps") and not result.reached_target
    if result.stop_reason == "no_improvement":
        assert result.steps == best["step"] + 6 * 10          # patience: 6 evaluations without a new best
    assert result.log_path == log_path

    delta = result.delta
    assert delta.kind == "trained" and delta.layers == LAYERS and delta.rank == RANK and delta.seed == 0
    assert all(A.shape == (WIDTH, RANK) and B.shape == (RANK, MLP_WIDTH) for A, B in zip(delta.A, delta.B))
    assert all(t.dtype == torch.float32 and t.device.type == "cpu" and not t.requires_grad for t in delta.A + delta.B)
    training = delta.info["training"]
    assert training["stop_reason"] == result.stop_reason and training["final"] == result.final
    assert training["lr"] == 3e-3 and training["n_validation"] == len(validation) and training["steps_run"] == result.steps
    json.dumps(delta.info)

    # The model comes back unedited, in eval mode, without hooks, with its flags as before.
    assert not model.training
    assert all(len(module._forward_hooks) == 0 for module in model.modules())
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    assert score_question(backend, wrong[0], str(backend.tokenizer_path)).is_correct is False

    # Applying the returned delta reproduces the reported numbers.
    tokenizer_path = str(backend.tokenizer_path)
    with applied(model, delta):
        fixed = sum(score_question(backend, q, tokenizer_path).is_correct for q in wrong)
        kept = sum(score_question(backend, q, tokenizer_path).is_correct for q in right)
        headline = [score_question(backend, q, tokenizer_path).is_correct for q in validation if q.section == "MEK"]
    assert (fixed, kept) == (result.final["n_wrong_fixed"], result.final["n_right_kept"])
    assert sum(headline) / len(headline) == result.final["validation"]["headline_accuracy"]


def test_snapshots_are_kept_the_first_time_each_fix_rate_is_reached(backend, tmp_path):
    log_path = tmp_path / "train.jsonl"
    result = train(backend, rule_task(backend), log_path, target_fix_rate=0.9)
    rows = read_log(log_path)

    assert result.reached_target and result.stop_reason == "target_fix_rate"
    assert rows[-1]["drill_wrong_fixed"] >= 0.9 and all(row["drill_wrong_fixed"] < 0.9 for row in rows[:-1])
    assert set(result.snapshots) == {"0.80", "0.85", "0.90"}
    for key, snapshot in result.snapshots.items():
        summary = snapshot.info["snapshot"]
        assert summary["step"] == next(row["step"] for row in rows if row["drill_wrong_fixed"] >= float(key))
        assert summary["kept_enough"] == (summary["drill_right_kept"] >= 0.95)
        on_disk = load_delta(tmp_path / f"train_fix{key.replace('.', '')}")
        assert all(torch.equal(a, b) for a, b in zip(on_disk.A + on_disk.B, snapshot.A + snapshot.B))
        assert on_disk.info == snapshot.info
    assert result.delta.info["training"]["snapshots"]["0.85"] == result.snapshots["0.85"].info["snapshot"]
    saved = save_snapshots(result.snapshots, tmp_path / "out" / "delta")
    assert saved["0.85"] == tmp_path / "out" / "delta_fix085" and (tmp_path / "out" / "delta_fix085.safetensors").exists()


def test_training_stops_when_validation_stops_improving(backend, tmp_path, monkeypatch):
    """With a learning rate of zero nothing ever beats the unedited model: stop after `patience` evaluations."""
    result = train(backend, rule_task(backend), tmp_path / "log.jsonl", lr=0.0, eval_every=2, patience=3)
    assert result.stop_reason == "no_improvement" and result.steps == 6 and result.final["step"] == 0
    assert all(float(B.abs().max()) == 0.0 for B in result.delta.B)      # the best "change" is no change
    assert [row["is_best"] for row in read_log(tmp_path / "log.jsonl")] == [True, False, False, False]


def test_training_stops_when_right_answers_are_lost(backend, tmp_path):
    """Early on the tiny model flips every answer to the new letter: three evaluations below 90% kept end the run."""
    result = train(backend, rule_task(backend), tmp_path / "log.jsonl", right_kept_floor=0.9, patience=100)
    rows = read_log(tmp_path / "log.jsonl")
    assert result.stop_reason == "right_kept_low" and result.steps < 200
    assert all(row["drill_right_kept"] < 0.9 for row in rows[-3:]) and rows[-4]["drill_right_kept"] >= 0.9


def test_validation_questions_are_never_trained_on(backend, tmp_path, monkeypatch):
    import bench.group_edit as group_edit

    wrong, right, validation = rule_task(backend)
    trained_on = []
    real_loss = group_edit.letter_cross_entropy
    monkeypatch.setattr(group_edit, "letter_cross_entropy",
                        lambda backend, question, ids=None: trained_on.append(question.id) or real_loss(backend, question, ids))
    train(backend, (wrong, right, validation), tmp_path / "log.jsonl", max_steps=20, shuffle=True)
    assert len(trained_on) == 20 * 8
    assert set(trained_on) == {q.id for q in wrong + right}
    assert not set(trained_on) & {q.id for q in validation}

    with pytest.raises(ValueError, match="validation questions are also trained on"):
        train(backend, (wrong, right, validation + [wrong[0]]), tmp_path / "log.jsonl")


def test_training_shows_shuffled_options_and_scores_the_original_order(backend, tmp_path, monkeypatch):
    import bench.group_edit as group_edit

    wrong, right, validation = rule_task(backend)
    by_id = {q.id: q for q in wrong + right + validation}
    trained_on, scored = [], []
    real_loss, real_probs = group_edit.letter_cross_entropy, backend.letter_probabilities
    monkeypatch.setattr(group_edit, "letter_cross_entropy",
                        lambda backend, question, ids=None: trained_on.append(question) or real_loss(backend, question, ids))
    monkeypatch.setattr(backend, "letter_probabilities",
                        lambda prompt, letters, question: scored.append(question) or real_probs(prompt, letters, question))

    train(backend, (wrong, right, validation), tmp_path / "log.jsonl", lr=1e-5, eval_every=6, max_steps=6, shuffle=True)

    assert len(trained_on) == 6 * 8
    wrong_ids, right_ids = {q.id for q in wrong}, {q.id for q in right}
    for step in range(6):                       # every batch: 4 wrong questions, then 4 right ones
        batch = [q.id for q in trained_on[step * 8:(step + 1) * 8]]
        assert all(i in wrong_ids for i in batch[:4]) and all(i in right_ids for i in batch[4:])
    assert {q.id for q in trained_on} == wrong_ids | right_ids            # every question is used
    for question in trained_on:                 # relabelled: the key still points at the correct option
        original = by_id[question.id]
        assert question.options[question.answer] == original.options[original.answer]
    assert sum(q.options != by_id[q.id].options for q in trained_on) > 40   # and really shuffled
    assert len({q.answer for q in trained_on}) == 5
    assert len(scored) == 2 * len(by_id)        # two evaluations, each of every drilled and validation question
    assert all(q.options == by_id[q.id].options and q.answer == by_id[q.id].answer for q in scored)


def test_training_is_repeatable_and_stops_at_max_steps(backend, tmp_path):
    task = rule_task(backend)
    runs = [train(backend, task, tmp_path / f"{i}.jsonl", lr=1e-5, eval_every=4, max_steps=6, seed=3, shuffle=True)
            for i in range(2)]
    assert all(torch.equal(a, b) for a, b in zip(runs[0].delta.A + runs[0].delta.B, runs[1].delta.A + runs[1].delta.B))
    assert runs[0].stop_reason == "max_steps" and runs[0].steps == 6
    assert [row["step"] for row in read_log(tmp_path / "0.jsonl")] == [0, 4, 6]   # the last step is always evaluated
    assert all(len(module._forward_hooks) == 0 for module in backend.model.modules())


def test_without_validation_the_drilled_questions_pick_the_checkpoint(backend, tmp_path):
    wrong, right, _ = rule_task(backend)
    result = train(backend, (wrong, right, None), tmp_path / "log.jsonl", max_steps=60)
    rows = read_log(tmp_path / "log.jsonl")
    assert all(row["validation"] is None for row in rows) and result.final["validation"] is None
    best = max(rows, key=lambda row: (row["drill_right_kept"] >= 0.95, row["n_wrong_fixed"], -row["step"]))
    assert result.final["step"] == best["step"]


def test_training_cleans_up_when_it_fails(backend, tmp_path, monkeypatch):
    task = rule_task(backend)
    name, weight = find_down_proj(backend.model, LAYERS[0])
    original = weight.detach().clone()
    calls = {"n": 0}
    real = backend.letter_probabilities

    def failing(prompt, letters, question):
        calls["n"] += 1
        if calls["n"] > 3:
            raise KeyError("boom")
        return real(prompt, letters, question)

    monkeypatch.setattr(backend, "letter_probabilities", failing)
    with pytest.raises(KeyError):
        train(backend, task, tmp_path / "log.jsonl")
    assert torch.equal(weight, original)
    assert all(len(module._forward_hooks) == 0 for module in backend.model.modules())
    assert all(p.requires_grad for p in backend.model.parameters())
    with pytest.raises(ValueError, match="no wrong questions"):
        train_group_delta(backend, [], task[1], tmp_path / "log.jsonl")


# ---------------------------------------------------------------- saving

def test_save_and_load_delta_round_trip(backend, tmp_path):
    delta = random_delta_like(some_delta(backend.model), seed=3)
    save_delta(delta, tmp_path / "deltas" / "trained")
    assert (tmp_path / "deltas" / "trained.json").exists() and (tmp_path / "deltas" / "trained.safetensors").exists()
    loaded = load_delta(tmp_path / "deltas" / "trained")
    assert all(torch.equal(a, b) for a, b in zip(loaded.A + loaded.B, delta.A + delta.B))
    for name in ("layers", "param_names", "rank", "kind", "seed", "info"):
        assert getattr(loaded, name) == getattr(delta, name)


@pytest.mark.parametrize("sharded", [False, True])
def test_saved_model_loads_and_matches_the_in_memory_merge(tokenizer_dir, tmp_path, sharded):
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tokenizer_dir))
    model = make_model(len(tokenizer), torch.bfloat16)
    src = tmp_path / "src"
    model.save_pretrained(str(src), **({"max_shard_size": "20KB"} if sharded else {}))
    tokenizer.save_pretrained(str(src))
    n_shards = len(list(src.glob("*.safetensors")))
    assert (n_shards > 1) == sharded
    delta = some_delta(model)

    out = save_merged_model(src, delta, tmp_path / "merged")

    assert out == tmp_path / "merged" and not (tmp_path / "merged.partial").exists()
    provenance = json.loads((out / "selma_group_edit.json").read_text())
    rewritten = {p.name for p in out.glob("*.safetensors") if not p.is_symlink()}
    linked = [p for p in out.glob("*.safetensors") if p.is_symlink()]
    assert rewritten == {entry["file"] for entry in provenance["rewritten"]}
    assert len(rewritten) + len(linked) == n_shards and (not sharded or linked)
    assert [entry["layer"] for entry in provenance["rewritten"]] == LAYERS
    assert all(entry["entries_changed"] > 0 for entry in provenance["rewritten"])
    if sharded:
        assert (out / "model.safetensors.index.json").read_text() == (src / "model.safetensors.index.json").read_text()
    for small in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        assert (out / small).is_file() and not (out / small).is_symlink()

    loaded = transformers.AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    with applied(model, delta):
        in_memory = {name: p.detach().clone() for name, p in model.named_parameters()}
    on_disk = dict(loaded.named_parameters())
    assert on_disk.keys() == in_memory.keys()
    for name, tensor in in_memory.items():     # bit for bit, the edited matrices and everything else
        assert on_disk[name].dtype == tensor.dtype and torch.equal(on_disk[name], tensor), name
    for name in delta.param_names:
        assert not torch.equal(on_disk[name], dict(model.named_parameters())[name])

    with pytest.raises(FileExistsError):
        save_merged_model(src, delta, tmp_path / "merged")


def test_saved_model_finds_weights_stored_under_reordered_names(tmp_path):
    """Gemma 3 files say `language_model.model.layers...`; the loaded model says `model.language_model.layers...`."""
    from safetensors.torch import load_file, save_file

    src = tmp_path / "src"
    src.mkdir()
    stored = {f"language_model.model.layers.{i}.mlp.down_proj.weight": torch.randn(WIDTH, MLP_WIDTH).to(torch.bfloat16)
              for i in range(3)}
    save_file(stored, str(src / "model.safetensors"), metadata={"format": "pt"})
    delta = GroupDelta(layers=[1, 2], param_names=[f"model.language_model.layers.{i}.mlp.down_proj.weight" for i in (1, 2)],
                       A=[torch.randn(WIDTH, 2) for _ in range(2)], B=[torch.randn(2, MLP_WIDTH) for _ in range(2)],
                       rank=2, kind="trained", seed=0, info={})
    out = save_merged_model(src, delta, tmp_path / "out")
    saved = load_file(str(out / "model.safetensors"))
    keys = list(stored)
    assert torch.equal(saved[keys[0]], stored[keys[0]])
    assert not torch.equal(saved[keys[1]], stored[keys[1]]) and not torch.equal(saved[keys[2]], stored[keys[2]])

    delta.param_names[1] = "model.language_model.layers.9.mlp.down_proj.weight"
    with pytest.raises(ValueError, match="cannot find"):
        save_merged_model(src, delta, tmp_path / "out2")
    assert not (tmp_path / "out2").exists()
