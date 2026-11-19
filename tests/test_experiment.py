"""The edit experiment driver end to end on dry backends, in a temporary project folder.

Two dry "full-precision" models (dry-a, dry-b: random answers with different seeds) stand in
for Qwen and Gemma, each with a dry "4-bit" entry. bench.experiment uses DryTools for them, so
no torch, GPU or llama.cpp is needed. The damage checks are replaced by a stub that writes a
plausible damage.json.
"""

import json
from dataclasses import asdict

import pytest

from bench import damage, experiment, paths, run
from bench.records import Answer, Question

DEV_DATES = ["2015-04-11", "2016-10-23"]
TEST_DATE = "2025-10-19"
PER_SECTION_PER_DATE = {"dev": 12, "test": 4}


def make_questions() -> tuple[list[Question], list[Question]]:
    dev, test = [], []
    for split, dates in (("dev", DEV_DATES), ("test", [TEST_DATE])):
        for date in dates:
            for section in ("ORD", "LAS", "MEK"):
                letters = ["A", "B", "C", "D", "E"] if section == "ORD" else ["A", "B", "C", "D"]
                for number in range(1, PER_SECTION_PER_DATE[split] + 1):
                    question = Question(
                        id=Question.make_id(date, 1, section, number), exam_date=date, exam_pass=1,
                        section=section, number=number, question=f"fråga {number}",
                        options={letter: f"alternativ {letter}" for letter in letters},
                        answer=letters[number % len(letters)],
                        passage="En text." if section == "LAS" else None,
                        passage_title="Titel" if section == "LAS" else None,
                    )
                    (dev if split == "dev" else test).append(question)
    return dev, test


MODELS_TOML = """
[dry-a]
backend = "random"
seed = 0

[dry-b]
backend = "random"
seed = 1

[dry-a-q4]
backend = "random"
seed = 0
parent = "dry-a"

[dry-b-q4]
backend = "random"
seed = 1
parent = "dry-b"
"""


def fake_damage_checks(backend, run_dir, data_dir=None):
    result = {"perplexity": backend.perplexity(["x"]), "n_perplexity_docs": 1,
              "loop_rate_percent": 0.0, "n_prompts": 1, "loop_rule": "test stub"}
    (run_dir / "damage.json").write_text(json.dumps(result), encoding="utf-8")
    return result


@pytest.fixture
def project(tmp_path, monkeypatch):
    dev, test = make_questions()
    exams = tmp_path / "exams"
    exams.mkdir()
    questions_file = exams / "questions.jsonl"
    questions_file.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n" for q in dev + test),
                              encoding="utf-8")
    split_file = exams / "split.json"
    split_file.write_text(json.dumps({
        "splits": {"dev": {"question_ids": [q.id for q in dev]},
                   "test": {"question_ids": [q.id for q in test]}},
        "questions_sha256": None,
    }), encoding="utf-8")
    models_file = tmp_path / "models.toml"
    models_file.write_text(MODELS_TOML, encoding="utf-8")

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "QUESTIONS", questions_file)
    monkeypatch.setattr(paths, "SPLIT", split_file)
    monkeypatch.setattr(paths, "MODELS_CONFIG", models_file)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(experiment, "EXPERIMENT_MODELS", ("dry-a", "dry-b"))
    monkeypatch.setattr(damage, "run_damage_checks", fake_damage_checks)
    return tmp_path


@pytest.fixture
def phase1(project):
    """The Phase 1 baselines the experiment starts from."""
    runs = {
        "dry-a_dev": run.run_model("dry-a", "dev"),
        "dry-b_dev": run.run_model("dry-b", "dev"),
        "dry-a-q4_test": run.run_model("dry-a-q4", "test"),
        "dry-b-q4_test": run.run_model("dry-b-q4", "test"),
    }
    return runs


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def answer(qid: str, section: str, correct: bool) -> Answer:
    return Answer(question_id=qid, section=section, correct_letter="A", chosen_letter="A" if correct else "B",
                  is_correct=correct, letter_probs={"A": 0.5, "B": 0.5}, prob_on_letters=1.0, prob_correct=0.5)


# ------------------------------------------------------------------ the fixed rules

def test_failure_rule_takes_wrong_answers_sorted_by_sha256_of_the_id():
    answers = [answer(f"2015-04-11/p1/{section}/{n}", section, correct=(n % 3 == 0))
               for section in ("ORD", "LAS", "MEK") for n in range(1, 13)]
    picked = experiment.pick_failure_cases(answers)

    assert [a.section for a in picked] == ["LAS"] * 4 + ["MEK"] * 4 + ["ORD"] * 2
    assert all(not a.is_correct for a in picked)
    for section, count in experiment.FAILURES_PER_SECTION.items():
        wrong = [a.question_id for a in answers if a.section == section and not a.is_correct]
        expected = sorted(wrong, key=experiment.sha256_hex)[:count]
        assert [a.question_id for a in picked if a.section == section] == expected


def test_failure_rule_takes_what_there_is_when_a_section_has_few_wrong_answers():
    answers = [answer("d/p1/ORD/1", "ORD", False), answer("d/p1/LAS/1", "LAS", True)]
    assert [a.question_id for a in experiment.pick_failure_cases(answers)] == ["d/p1/ORD/1"]


def test_select_failures_uses_the_latest_full_dev_run_and_writes_both_files(project, phase1):
    run.run_model("dry-a", "dev", limit=3)   # a later quick check must not count as the baseline
    rows = experiment.select_failures("dry-a")

    assert len(rows) == 10
    assert {row["baseline_run"] for row in rows} == {experiment.rel(phase1["dry-a_dev"])}
    baseline_answers = {a["question_id"]: a for a in read_jsonl(phase1["dry-a_dev"] / "answers.jsonl")}
    for row in rows:
        assert baseline_answers[row["question_id"]]["is_correct"] is False
        assert row["chosen_letter"] == baseline_answers[row["question_id"]]["chosen_letter"]
        assert row["question"]["id"] == row["question_id"]
    markdown = (project / "experiments" / "dry-a" / "failure_cases.md").read_text(encoding="utf-8")
    assert "English explanation" in markdown
    assert markdown.count(experiment.EXPLANATION_PLACEHOLDER) == 10


def test_select_failures_without_a_baseline_says_what_to_run(project):
    with pytest.raises(FileNotFoundError, match="bench run dry-a --split dev"):
        experiment.select_failures("dry-a")


def test_dev_subset_is_stratified_deterministic_and_excludes_failures_of_both_models(project, phase1):
    first = [q.id for q in experiment.dev_subset("dry-a", per_section=5)]
    path = project / "experiments" / "dev_subset.json"
    assert path.exists()

    failures = {row["question_id"] for model in ("dry-a", "dry-b") for row in experiment.select_failures(model)}
    assert not failures & set(first)
    assert len(first) == 15
    assert [qid.split("/")[2] for qid in first] == ["ORD"] * 5 + ["LAS"] * 5 + ["MEK"] * 5

    assert [q.id for q in experiment.dev_subset("dry-b")] == first   # the other model reads the same file
    path.unlink()
    assert [q.id for q in experiment.dev_subset("dry-b", per_section=5)] == first   # rebuilt identically


# ------------------------------------------------------------------ the whole flow

def test_whole_flow_on_dry_backends(project, phase1):
    experiment.run_all("dry-a", with_combined=True, with_control_q4=True)

    model_dir = project / "experiments" / "dry-a"
    searches = [json.loads(p.read_text()) for p in sorted((model_dir / "edits").glob("*/search.json"))]
    assert len(searches) == 10
    fixed = [s for s in searches if s["fixed"]]
    assert fixed and len(fixed) < 10   # the dry tools fix most cases but not all

    rows = read_jsonl(model_dir / "results.jsonl")
    kinds = [row["kind"] for row in rows]
    assert kinds.count("targeted") == 10
    assert kinds.count("control") == len(fixed)
    assert kinds.count("combined") == kinds.count("combined_control") == (1 if len(fixed) >= 2 else 0)

    for row in rows:
        if row["kind"] == "targeted" and row["fixed"]:
            assert row["target"]["full"]["before"]["is_correct"] is False
            assert row["target"]["full"]["after"]["is_correct"] is True   # the dry edit fixes its target
            assert row["test"]["verdict"] in ("IMPROVEMENT", "NO IMPROVEMENT")
            assert row["test"]["control"] == "control_test"
            assert row["test"]["damage"] is not None
            assert row["dev_subset"]["indicative_only"] is True
            assert row["q4_test"]["baseline"] == phase1["dry-a-q4_test"].name
            assert row["q4_test"]["control"] == "control_q4_test"
            assert row["survives_quantization"] is False   # the dry 4-bit model does not carry the edit
            assert set(row["test"]["letter_shift"]["overall"]) == set("ABCDE")
            assert row["alpha_fraction"] == pytest.approx(0.1)
        if row["kind"] == "control":
            assert row["edit"]["kind"] == "random"
            assert row["target"]["full"]["after"]["is_correct"] is False
        if row["kind"] == "targeted" and not row["fixed"]:
            assert row["status"] == "could_not_fix"
            assert row["target"]["full"]["after"] is None
        if row["kind"] == "combined":
            assert all(entry["after"]["is_correct"] for entry in row["target"]["full"].values())

    # Damage only on the test runs.
    one_case = model_dir / "edits" / experiment.case_slug(fixed[0]["question_id"])
    assert (one_case / "targeted_test" / "damage.json").exists()
    assert not (one_case / "targeted_dev_subset" / "damage.json").exists()
    assert not (one_case / "targeted_target" / "damage.json").exists()
    info = json.loads((one_case / "targeted_test" / "run.json").read_text())
    assert info["model"] == f"dry-a+edit[{fixed[0]['question_id']}]"
    assert info["experiment"]["edits"][0]["kind"] == "targeted"

    # The temporary edited models and 4-bit files are gone.
    assert not any(path.is_file() for path in (project / "work").rglob("*"))


def test_second_run_does_no_work(project, phase1, monkeypatch):
    experiment.run_all("dry-a", with_combined=True, with_control_q4=True)

    calls = []
    real_run_with_backend = experiment.run_with_backend

    def counting_run(*args, **kwargs):
        calls.append(kwargs.get("run_dir"))
        return real_run_with_backend(*args, **kwargs)

    def counting_make_backend(name):
        calls.append(name)
        raise AssertionError("no model should be loaded when every step is done")

    monkeypatch.setattr(experiment, "run_with_backend", counting_run)
    monkeypatch.setattr(experiment, "make_backend", counting_make_backend)
    experiment.run_all("dry-a", with_combined=True, with_control_q4=True)
    assert calls == []


def test_an_unfinished_run_folder_is_redone(project, phase1):
    experiment.run_all("dry-a")
    full_test = project / "experiments" / "dry-a" / "baseline" / "full_test"
    info = json.loads((full_test / "run.json").read_text())
    del info["finished_at"]
    (full_test / "run.json").write_text(json.dumps(info))

    experiment.run_edits("dry-a")
    assert experiment.run_is_complete(full_test, need_damage=True)


def test_q4_entry_is_found_from_the_parent_field(project):
    assert experiment.q4_model_for("dry-a") == "dry-a-q4"
    with pytest.raises(ValueError, match="parent = 'dry-a-q4'"):
        experiment.q4_model_for("dry-a-q4")


# ------------------------------------------------------------------ the real-edit glue, on a tiny torch model

def tiny_model_and_edits():
    torch = pytest.importorskip("torch")
    from bench.edit import Edit

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Module()
            self.mlp.down_proj = torch.nn.Linear(6, 4, bias=False)

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([Block(), Block()])
    generator = torch.Generator().manual_seed(0)
    edits = []
    for alpha in (0.3, -0.2):
        u = torch.randn(4, generator=generator)
        v = torch.randn(6, generator=generator)
        edits.append(Edit(layer=1, param_name="layers.1.mlp.down_proj.weight", u=u / u.norm(), v=v / v.norm(),
                          alpha=alpha, alpha_fraction=alpha, kind="targeted", question_id=f"q{alpha}", seed=None))
    return torch, model, edits


def test_summed_edits_are_applied_together_and_restored_exactly():
    torch, model, edits = tiny_model_and_edits()
    weight = model.get_parameter("layers.1.mlp.down_proj.weight")
    original = weight.detach().clone()
    expected = original + sum(e.alpha * torch.outer(e.u, e.v) for e in edits)

    with experiment.summed_edits_applied(model, edits) as change:
        assert torch.allclose(weight, expected, atol=1e-6)
        assert change["realized_fraction"] == pytest.approx(1.0, abs=1e-5)
    assert torch.equal(weight, original)


def test_edit_tools_apply_one_edit_with_bench_edit_and_restore():
    torch, model, edits = tiny_model_and_edits()

    class FakeBackend:
        pass

    backend = FakeBackend()
    backend.model = model
    weight = model.get_parameter("layers.1.mlp.down_proj.weight")
    original = weight.detach().clone()
    tools = experiment.EditTools()
    with tools.applied(backend, edits[:1]) as changes:
        assert not torch.equal(weight, original)
        assert changes[0]["realized_fraction"] == pytest.approx(1.0, abs=1e-5)
    assert torch.equal(weight, original)
    assert tools.reference_norm(backend, 1) == pytest.approx(float(torch.linalg.matrix_norm(original, ord=2)))
