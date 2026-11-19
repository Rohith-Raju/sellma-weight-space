"""The group edit driver end to end on dry backends, in a temporary project folder.

A dry "full-precision" model (gemma-dry: random answers) stands in for Gemma, with a dry "4-bit"
entry. bench.group_edit needs torch and a real model, so a stand-in module is put in its place
(sys.modules): its "trained change" makes the dry model answer the drill wrong questions
correctly, and in "generalising" mode also about half of all other questions it was not trained
on; its random change does nothing. The damage checks are replaced by a stub.
"""

import hashlib
import json
import sys
import types
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest

from bench import cli, damage, group_experiment as ge, paths, run
from bench.records import Question

# Nine dev papers: 3 hidden, 3 validation, 3 drill.
DEV_DATES = [f"20{year}-04-0{day}" for year, day in
             ((13, 1), (14, 2), (15, 3), (16, 4), (17, 5), (18, 6), (19, 7), (20, 8), (21, 9))]
TEST_DATE = "2025-10-19"
EXTRA_DATE = "2024-10-20"
PER_SECTION = {"dev": 6, "test": 4, "gemma_extra": 4}

MODELS_TOML = """
[gemma-dry]
backend = "random"
seed = 0

[gemma-dry-q4]
backend = "random"
seed = 0
parent = "gemma-dry"
"""

# The 23 real dev exam dates and what the rule makes of them (also checked against
# exams/split.json by hand when the experiment was set up: 480 hidden, 150 validation, 720 drill).
REAL_DEV_DATES = ["2013-10-26", "2014-04-05", "2014-10-25", "2015-03-28", "2015-10-24", "2016-04-09",
                  "2016-10-29", "2017-04-01", "2017-10-21", "2018-04-14", "2018-10-21", "2019-04-06",
                  "2019-10-20", "2020-10-25", "2021-03-13", "2021-05-08", "2021-10-24", "2022-03-12",
                  "2022-05-07", "2022-10-23", "2023-03-25", "2023-10-22", "2024-04-13"]
REAL_HIDDEN = ["2015-03-28", "2016-04-09", "2017-10-21", "2018-10-21", "2019-10-20", "2020-10-25",
               "2021-05-08", "2022-10-23"]
REAL_VALIDATION = ["2018-04-14", "2022-03-12", "2023-10-22"]


# ------------------------------------------------------------------ the stand-in for bench.group_edit

def make_fake_group_edit(generalise: bool) -> types.ModuleType:
    module = types.ModuleType("bench.group_edit")
    module.calls = []

    @dataclass
    class GroupDelta:
        layers: list
        param_names: list
        A: list
        B: list
        rank: int
        kind: str
        seed: int | None
        info: dict = field(default_factory=dict)

    @dataclass
    class TrainResult:
        delta: GroupDelta
        reached_target: bool
        steps: int
        final: dict
        log_path: Path

    def train_group_delta(backend, wrong, right, log_path, validation=None, max_steps=600, **kwargs):
        module.calls.append({"wrong": [q.id for q in wrong], "right": [q.id for q in right],
                             "validation": [q.id for q in (validation or [])], "max_steps": max_steps})
        Path(log_path).write_text("".join(
            json.dumps({"step": step, "validation_headline_accuracy": 0.3 + 0.01 * min(step, 50) / 25,
                        "drill_wrong_fixed": min(1.0, step / 50), "drill_right_kept": 1.0 - step / 1000}) + "\n"
            for step in range(0, 101, 25)), encoding="utf-8")
        delta = GroupDelta(layers=[1, 2], param_names=["a", "b"], A=[], B=[], rank=8, kind="trained", seed=0,
                           info={"fix_ids": [q.id for q in wrong], "generalise": generalise,
                                 "training": {"best_step": 50, "stopped_because": "validation stopped improving",
                                              "steps_run": 100}})
        return TrainResult(delta=delta, reached_target=True, steps=100, log_path=Path(log_path),
                           final={"step": 50, "drill_wrong_fixed": 1.0, "drill_right_kept": 0.95})

    def random_delta_like(delta, seed):
        return GroupDelta(layers=list(delta.layers), param_names=list(delta.param_names), A=[], B=[],
                          rank=delta.rank, kind="random", seed=seed, info={})

    def fixes(delta, question_id):
        if delta.kind != "trained":
            return False
        if question_id in delta.info["fix_ids"]:
            return True
        return delta.info["generalise"] and hashlib.sha256(question_id.encode()).hexdigest()[0] in "01234567"

    @contextmanager
    def applied(model, delta):
        backend = model   # a dry backend has no weights; the driver passes the backend itself
        original = backend.letter_probabilities

        def patched(prompt, letters, question):
            if fixes(delta, question.id):
                return {letter: 0.9 if letter == question.answer else 0.01 for letter in letters}
            return original(prompt, letters, question)

        backend.letter_probabilities = patched
        try:
            yield [{"realized_fraction": 1.0}]
        finally:
            del backend.letter_probabilities

    def save_delta(delta, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".json").write_text(json.dumps(asdict(delta)), encoding="utf-8")

    def load_delta(path):
        return GroupDelta(**json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8")))

    def save_merged_model(src_dir, delta, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True)
        (out_dir / "config.json").write_text("{}", encoding="utf-8")
        return out_dir

    for name, value in list(locals().items()):
        if not name.startswith("_") and name not in ("module", "generalise"):
            setattr(module, name, value)
    return module


@pytest.fixture
def generalising(monkeypatch):
    fake = make_fake_group_edit(generalise=True)
    monkeypatch.setitem(sys.modules, "bench.group_edit", fake)
    return fake


@pytest.fixture
def memorising(monkeypatch):
    fake = make_fake_group_edit(generalise=False)
    monkeypatch.setitem(sys.modules, "bench.group_edit", fake)
    return fake


# ------------------------------------------------------------------ the temporary project

def make_questions() -> dict[str, list[Question]]:
    splits: dict[str, list[Question]] = {"dev": [], "test": [], "gemma_extra": []}
    for split, dates in (("dev", DEV_DATES), ("test", [TEST_DATE]), ("gemma_extra", [EXTRA_DATE])):
        for date in dates:
            for section in ("ORD", "LAS", "MEK"):
                letters = ["A", "B", "C", "D", "E"] if section == "ORD" else ["A", "B", "C", "D"]
                for number in range(1, PER_SECTION[split] + 1):
                    splits[split].append(Question(
                        id=Question.make_id(date, 1, section, number), exam_date=date, exam_pass=1,
                        section=section, number=number, question=f"fråga {number}",
                        options={letter: f"alternativ {letter}" for letter in letters},
                        answer=letters[number % len(letters)],
                        passage="En text." if section == "LAS" else None,
                        passage_title="Titel" if section == "LAS" else None,
                    ))
    return splits


def fake_damage_checks(backend, run_dir, data_dir=None):
    result = {"perplexity": backend.perplexity(["x"]), "n_perplexity_docs": 1,
              "loop_rate_percent": 0.0, "n_prompts": 1, "loop_rule": "test stub"}
    (run_dir / "damage.json").write_text(json.dumps(result), encoding="utf-8")
    return result


@pytest.fixture
def project(tmp_path, monkeypatch):
    splits = make_questions()
    exams = tmp_path / "exams"
    exams.mkdir()
    questions_file = exams / "questions.jsonl"
    questions_file.write_text("".join(json.dumps(asdict(q), ensure_ascii=False) + "\n"
                                      for questions in splits.values() for q in questions), encoding="utf-8")
    split_file = exams / "split.json"
    split_file.write_text(json.dumps({
        "splits": {name: {"question_ids": [q.id for q in questions]} for name, questions in splits.items()},
        "questions_sha256": None,
    }), encoding="utf-8")
    models_file = tmp_path / "models.toml"
    models_file.write_text(MODELS_TOML, encoding="utf-8")

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "QUESTIONS", questions_file)
    monkeypatch.setattr(paths, "SPLIT", split_file)
    monkeypatch.setattr(paths, "MODELS_CONFIG", models_file)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(damage, "run_damage_checks", fake_damage_checks)
    return tmp_path


@pytest.fixture
def phase1(project):
    """The Phase 1 baselines the experiment starts from."""
    return {
        "dev": run.run_model("gemma-dry", "dev"),
        "q4_test": run.run_model("gemma-dry-q4", "test"),
        "q4_gemma_extra": run.run_model("gemma-dry-q4", "gemma_extra"),
    }


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def model_dir(project: Path) -> Path:
    return project / "experiments" / "group" / "gemma-dry"


# ------------------------------------------------------------------ the split rule

def test_split_rule_on_the_real_dev_dates():
    dates = ge.split_dates(REAL_DEV_DATES)
    assert dates["hidden"] == REAL_HIDDEN
    assert dates["validation"] == REAL_VALIDATION
    assert len(dates["drill"]) == 12
    assert sorted(dates["hidden"] + dates["validation"] + dates["drill"]) == REAL_DEV_DATES


def test_split_rule_is_sha256_order():
    ordered = sorted(DEV_DATES, key=lambda d: hashlib.sha256(d.encode()).hexdigest())
    dates = ge.split_dates(list(reversed(DEV_DATES)))   # input order does not matter
    assert dates["hidden"] == sorted(ordered[:3])
    assert dates["validation"] == sorted(ordered[3:6])
    assert dates["drill"] == sorted(ordered[6:])


def test_make_split_groups_follow_the_baseline_and_are_written_once(project, phase1):
    data = ge.make_split("gemma-dry")
    baseline = {a["question_id"]: a["is_correct"]
                for a in map(json.loads, (phase1["dev"] / "answers.jsonl").read_text().splitlines())}
    ids = data["question_ids"]
    assert sum(data["counts"].values()) == len(baseline) == 9 * 18
    for part in ge.PARTS:
        assert all(not baseline[qid] for qid in ids[f"{part}_wrong"])
        assert all(baseline[qid] for qid in ids[f"{part}_right"])
        dates = {qid.split("/")[0] for qid in ids[f"{part}_wrong"] + ids[f"{part}_right"]}
        assert dates == set(data["dates"][part])
        assert data["counts"][f"{part}_wrong"] == len(ids[f"{part}_wrong"])
    for name in ge.GROUPS:
        assert sum(data["counts_by_section"][name].values()) == data["counts"][name]

    path = model_dir(project) / "split.json"
    before = path.read_text()
    run.run_model("gemma-dry", "dev")   # a newer baseline run does not change a split already written
    assert ge.make_split("gemma-dry") == data
    assert path.read_text() == before


# ------------------------------------------------------------------ the whole flow

def test_whole_flow_with_a_change_that_generalises(project, phase1, generalising):
    ge.run_all("gemma-dry", with_control_q4=True)
    folder = model_dir(project)
    split = read_json(folder / "split.json")

    # Training saw the drill papers, stopped by the validation papers, and never saw a hidden question.
    call, = generalising.calls
    assert call["wrong"] == split["question_ids"]["drill_wrong"]
    assert call["right"] == split["question_ids"]["drill_right"]
    assert sorted(call["validation"]) == sorted(split["question_ids"]["validation_wrong"]
                                                + split["question_ids"]["validation_right"])
    hidden = set(split["question_ids"]["hidden_wrong"] + split["question_ids"]["hidden_right"])
    assert not hidden & set(call["wrong"] + call["right"] + call["validation"])

    for condition in ge.CONDITIONS:
        for name in ("hidden", "drill", "validation", "test", "gemma_extra"):
            assert ge.run_is_complete(folder / condition / name), (condition, name)
    assert (folder / "trained" / "test" / "damage.json").exists()
    assert not (folder / "trained" / "hidden" / "damage.json").exists()
    for condition in ge.EDITED:
        for name in ("q4_hidden", "q4_test", "q4_gemma_extra"):
            assert ge.run_is_complete(folder / condition / name), (condition, name)
    assert ge.run_is_complete(folder / "baseline" / "q4_hidden")
    assert not (folder / "baseline" / "q4_test").exists()   # the Phase 1 4-bit run is used instead
    info = read_json(folder / "trained" / "hidden" / "run.json")
    assert info["model"] == "gemma-dry+group[trained]"
    assert info["experiment"]["delta"]["kind"] == "trained"
    assert not any(path.is_file() for path in (project / "work").rglob("*"))

    results = read_json(folder / "results.json")
    full = results["full"]
    assert full["baseline_check"]["hidden"]["disagreements"] == 0
    drill = full["trained"]["drill"]
    assert drill["fixed"] == drill["n_wrong"] and drill["broken"] == 0
    assert full["control"]["hidden"]["fixed"] == 0
    key = full["trained_vs_control"]["hidden_wrong_fixed"]
    assert key["change"] > 0 and key["ci"][0] > 0
    assert full["trained"]["test_vs_baseline"]["control"] == "test"
    assert full["trained"]["test_vs_baseline"]["damage"] is not None
    assert full["trained"]["gemma_extra_vs_baseline"] is not None
    assert set(full["trained"]["hidden_by_section"]) == {"ORD", "LAS", "MEK"}
    assert results["q4"]["trained"]["test_vs_baseline"]["baseline"] == phase1["q4_test"].name
    # The dry 4-bit model does not carry the change, so at 4-bit there is nothing to see.
    assert results["q4"]["trained_vs_control"]["hidden_wrong_fixed"]["change"] == 0
    assert results["training"]["chosen_step"] == 50
    assert results["training_curve"]["validation_key"] == "validation_headline_accuracy"
    assert len(results["training_curve"]["points"]) == 5

    markdown = (folder / "results.md").read_text()
    full_part, q4_part = results["verdict"].split("After compression to 4-bit")
    assert "does generalise" in full_part
    assert "does generalise" not in q4_part and "includes zero" in q4_part
    assert "| 50 (chosen) |" in markdown
    assert "**Hidden wrong: fixed**" in markdown
    assert "Validation papers: " in markdown


def test_a_change_that_only_memorises_is_never_called_a_gain(project, phase1, memorising):
    ge.run_all("gemma-dry")
    results = read_json(model_dir(project) / "results.json")
    key = results["full"]["trained_vs_control"]["hidden_wrong_fixed"]
    assert key["change"] == 0
    verdict = results["verdict"]
    assert "includes zero" in verdict
    assert "does generalise" not in verdict and "real net gain" not in verdict
    assert results["full"]["trained"]["test_vs_baseline"]["verdict"] == "NO IMPROVEMENT"
    assert "NO IMPROVEMENT" in verdict
    assert "hidden" not in results["q4"]["control"]   # no 4-bit control without --with-control-q4
    assert "There is no random-change run" in verdict


def test_second_run_does_no_work(project, phase1, generalising, monkeypatch):
    ge.run_all("gemma-dry", with_control_q4=True)
    calls = []

    def counting_run(*args, **kwargs):
        calls.append(kwargs.get("run_dir"))

    def no_backend(name):
        raise AssertionError(f"no model should be loaded when every step is done (asked for {name})")

    monkeypatch.setattr(ge, "run_with_backend", counting_run)
    monkeypatch.setattr(ge, "make_backend", no_backend)
    ge.run_all("gemma-dry", with_control_q4=True)
    assert calls == []
    assert len(generalising.calls) == 1   # trained once


def test_an_unfinished_run_folder_is_redone(project, phase1, generalising):
    ge.run_all("gemma-dry")
    hidden = model_dir(project) / "trained" / "hidden"
    info = read_json(hidden / "run.json")
    del info["finished_at"]
    (hidden / "run.json").write_text(json.dumps(info))

    ge.evaluate("gemma-dry")
    assert ge.run_is_complete(hidden)
    assert len(generalising.calls) == 1


def test_steps_after_split_need_training_first(project, phase1, generalising):
    with pytest.raises(FileNotFoundError, match="bench group train --model gemma-dry"):
        ge.evaluate("gemma-dry")


def test_cli_runs_a_step(project, phase1, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "group", "split", "--model", "gemma-dry"])
    cli.main()
    assert (model_dir(project) / "split.json").exists()


# ------------------------------------------------------------------ the verdict wording

def hidden_numbers(fixed: int, broken: int, net_ci: list[float]) -> dict:
    return {"n_wrong": 100, "fixed": fixed, "fixed_share": fixed / 100, "n_right": 300, "kept": 300 - broken,
            "kept_share": (300 - broken) / 300, "broken": broken, "net_questions": fixed - broken,
            "net": {"accuracy_change": (fixed - broken) / 400, "ci": net_ci}}


@pytest.mark.parametrize("key_ci, net_ci, expected, forbidden", [
    ([-0.01, 0.05], [-0.01, 0.02], ["includes zero", "cannot be told apart"], ["does generalise", "real net gain"]),
    ([0.02, 0.09], [0.005, 0.03], ["does generalise", "real net gain"], ["includes zero"]),
    ([-0.08, -0.01], [-0.04, -0.01], ["does not generalise", "real net loss"], ["does generalise", "real net gain"]),
])
def test_verdict_follows_the_confidence_interval(key_ci, net_ci, expected, forbidden):
    numbers = {"trained": {"hidden": hidden_numbers(12, 5, net_ci)},
               "control": {"hidden": hidden_numbers(10, 4, [-0.02, 0.03])},
               "trained_vs_control": {"hidden_wrong_fixed": {"change": sum(key_ci) / 2, "ci": key_ci}}}
    text = " ".join(ge.precision_sentences(numbers, "At full precision"))
    for words in expected:
        assert words in text
    for words in forbidden:
        assert words not in text
