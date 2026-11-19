"""run_model end to end with the dry backends, on a tiny question set in a temporary folder."""

import json
import sys
import types
from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from bench import paths, run
from bench.records import Question
from bench.run import score_answer

SECTIONS = ["ORD", "LAS", "MEK", "ORD", "MEK", "LAS"]
ANSWERS = ["A", "B", "C", "D", "D", "A"]   # MEK has options A-D only, so its answers must be too


def make_questions() -> list[Question]:
    questions = []
    for number, (section, answer) in enumerate(zip(SECTIONS, ANSWERS), start=1):
        letters = ["A", "B", "C", "D", "E"] if section == "ORD" else ["A", "B", "C", "D"]
        questions.append(Question(
            id=Question.make_id("2025-10-19", 2, section, number),
            exam_date="2025-10-19",
            exam_pass=2,
            section=section,
            number=number,
            question=f"fråga {number}",
            options={letter: f"alternativ {letter}{number}" for letter in letters},
            answer=answer,
            passage="En text." if section == "LAS" else None,
            passage_title="Titel" if section == "LAS" else None,
        ))
    return questions


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project: questions, a split, a models.toml, and redirected run folders."""
    questions = make_questions()
    exams = tmp_path / "exams"
    exams.mkdir()
    questions_file = exams / "questions.jsonl"
    questions_file.write_text(
        "\n".join(json.dumps(_as_dict(q), ensure_ascii=False) for q in questions) + "\n", encoding="utf-8")

    split_file = exams / "split.json"
    split_file.write_text(json.dumps({
        "splits": {
            "dev": {"question_ids": [q.id for q in questions]},
            "test": {"question_ids": [questions[0].id]},
        },
        "questions_sha256": None,   # no frozen checksum, so no warning
    }), encoding="utf-8")

    models_file = tmp_path / "models.toml"
    models_file.write_text(
        '[dry-oracle]\nbackend = "oracle"\n\n[dry-random]\nbackend = "random"\nseed = 0\n',
        encoding="utf-8")

    monkeypatch.setattr(paths, "ROOT", tmp_path)
    monkeypatch.setattr(paths, "QUESTIONS", questions_file)
    monkeypatch.setattr(paths, "SPLIT", split_file)
    monkeypatch.setattr(paths, "MODELS_CONFIG", models_file)
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    return tmp_path


@pytest.fixture
def stubs(monkeypatch):
    """Stand-ins for bench.damage and bench.report, which are written by other agents."""
    calls = {"damage": [], "report": []}
    damage = types.ModuleType("bench.damage")
    damage.run_damage_checks = lambda backend, run_dir: calls["damage"].append(run_dir) or {}
    report = types.ModuleType("bench.report")
    report.summarize_run = lambda run_dir: calls["report"].append(run_dir) or {}
    monkeypatch.setitem(sys.modules, "bench.damage", damage)
    monkeypatch.setitem(sys.modules, "bench.report", report)
    return calls


def _as_dict(question: Question) -> dict:
    return asdict(question)


def read_answers(run_dir):
    with open(run_dir / "answers.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_oracle_run_scores_every_question_correctly(project, stubs):
    run_dir = run.run_model("dry-oracle", "dev")

    answers = read_answers(run_dir)
    assert len(answers) == len(SECTIONS)
    assert all(answer["is_correct"] for answer in answers)
    assert [answer["chosen_letter"] for answer in answers] == ANSWERS
    assert [answer["section"] for answer in answers] == SECTIONS
    assert all(answer["prob_on_letters"] == pytest.approx(1.0) for answer in answers)
    assert all(answer["prob_correct"] == pytest.approx(0.97) for answer in answers)


def test_answer_letter_probs_cover_exactly_the_option_letters(project, stubs):
    run_dir = run.run_model("dry-random", "dev")

    for answer, question in zip(read_answers(run_dir), make_questions()):
        assert sorted(answer["letter_probs"]) == sorted(question.options)
        assert answer["question_id"] == question.id
        assert answer["correct_letter"] == question.answer


def test_run_json_records_what_was_run(project, stubs):
    run_dir = run.run_model("dry-oracle", "dev")

    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["model"] == "dry-oracle"
    assert info["backend"] == "oracle"
    assert info["split"] == "dev"
    assert info["model_config"] == {"backend": "oracle"}
    assert info["describe"]["backend"] == "oracle"
    assert len(info["questions_sha256"]) == 64
    assert len(info["split_sha256"]) == 64
    assert info["n_questions"] == len(SECTIONS)
    assert info["started_at"] and info["finished_at"]
    assert isinstance(info["command"], str)


def test_run_folder_name_has_time_model_and_split(project, stubs):
    run_dir = run.run_model("dry-oracle", "dev")
    assert run_dir.parent == paths.RUNS
    assert run_dir.name.endswith("_dry-oracle_dev")
    assert run_dir.name[:8].isdigit()


def test_two_runs_in_the_same_second_get_separate_folders(project, stubs, monkeypatch):
    class FrozenClock:
        @staticmethod
        def now(tz=None):
            return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    monkeypatch.setattr(run, "datetime", FrozenClock)
    first = run.run_model("dry-oracle", "dev", limit=1)
    second = run.run_model("dry-oracle", "dev", limit=1)
    assert first != second
    assert len(read_answers(second)) == 1


def test_limit_takes_the_first_questions_of_the_split(project, stubs):
    run_dir = run.run_model("dry-oracle", "dev", limit=2)
    answers = read_answers(run_dir)
    assert [answer["question_id"] for answer in answers] == [q.id for q in make_questions()[:2]]
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["n_questions"] == 2


def test_damage_and_report_run_unless_skipped(project, stubs):
    run_dir = run.run_model("dry-oracle", "dev")
    assert stubs["damage"] == [run_dir]
    assert stubs["report"] == [run_dir]


def test_skip_damage_does_not_call_damage_or_report(project, stubs):
    run.run_model("dry-oracle", "dev", skip_damage=True)
    assert stubs["damage"] == []
    assert stubs["report"] == []


def test_split_with_other_name_uses_its_own_ids(project, stubs):
    run_dir = run.run_model("dry-oracle", "test")
    answers = read_answers(run_dir)
    assert [answer["question_id"] for answer in answers] == [make_questions()[0].id]


def test_unknown_split_fails_with_available_names(project, stubs):
    with pytest.raises(KeyError, match="available splits: dev, test"):
        run.run_model("dry-oracle", "gemma_extra")


def test_missing_split_file_fails_with_a_clear_message(project, stubs, monkeypatch):
    monkeypatch.setattr(paths, "SPLIT", project / "exams" / "nope.json")
    with pytest.raises(FileNotFoundError, match="bench split"):
        run.run_model("dry-oracle", "dev")


def test_backend_is_closed_even_when_scoring_fails(project, stubs, monkeypatch):
    closed = []

    class FailingBackend:
        def letter_probabilities(self, prompt, letters, question):
            raise RuntimeError("model broke")

        def describe(self):
            return {"backend": "failing"}

        def close(self):
            closed.append(True)

    monkeypatch.setattr(run, "make_backend", lambda name: FailingBackend())
    with pytest.raises(RuntimeError, match="model broke"):
        run.run_model("dry-oracle", "dev")
    assert closed == [True]


def test_score_answer_normalises_over_option_letters():
    question = make_questions()[0]   # ORD, answer "A", options A-E
    answer = score_answer(question, {"A": 0.2, "B": 0.6, "C": 0.1})

    assert answer.chosen_letter == "B"
    assert answer.is_correct is False
    assert answer.prob_on_letters == pytest.approx(0.9)
    assert answer.prob_correct == pytest.approx(0.2 / 0.9)
    assert answer.letter_probs == {"A": 0.2, "B": 0.6, "C": 0.1, "D": 0.0, "E": 0.0}


def test_score_answer_with_no_letter_mass_has_zero_probability():
    question = make_questions()[0]
    answer = score_answer(question, {})
    assert answer.prob_on_letters == 0.0
    assert answer.prob_correct == 0.0


def test_score_answer_with_no_letter_mass_is_never_correct():
    question = make_questions()[0]   # answer "A", which is also the tie-break letter
    answer = score_answer(question, {"A": 0.0, "B": 0.0})
    assert answer.is_correct is False


def test_real_model_without_tokenizer_is_refused_before_loading(project, stubs, monkeypatch):
    (project / "models.toml").write_text('[real-no-tokenizer]\nbackend = "hf"\npath = "models/x"\n',
                                         encoding="utf-8")
    loaded = []
    monkeypatch.setattr(run, "make_backend", lambda name: loaded.append(name))
    with pytest.raises(ValueError, match="no tokenizer folder"):
        run.run_model("real-no-tokenizer", "dev")
    assert loaded == []


def test_run_with_backend_refuses_real_backend_without_tokenizer(project, stubs):
    from bench.backends.dry import OracleBackend

    with pytest.raises(ValueError, match="no tokenizer folder"):
        run.run_with_backend(OracleBackend(), "label", "dev", make_questions(), None, backend_type="llamacpp")
    assert not paths.RUNS.exists()


def test_run_with_backend_writes_into_the_given_folder_and_keeps_backend_open(project, stubs):
    class CountingOracle:
        closed = False

        def __init__(self):
            from bench.backends.dry import OracleBackend
            self.inner = OracleBackend()

        def letter_probabilities(self, prompt, letters, question):
            return self.inner.letter_probabilities(prompt, letters, question)

        def describe(self):
            return {"backend": "oracle"}

        def close(self):
            self.closed = True

    backend = CountingOracle()
    questions = make_questions()[:2]
    run_dir = run.run_with_backend(backend, "my-label", "dev", questions, None, skip_damage=True,
                                   extra_info={"experiment": {"variant": "targeted"}},
                                   run_dir=project / "somewhere" / "targeted_test")
    assert run_dir == project / "somewhere" / "targeted_test"
    assert backend.closed is False
    info = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert info["model"] == "my-label"
    assert info["backend"] == "oracle"
    assert info["experiment"] == {"variant": "targeted"}
    assert info["finished_at"]
    assert len(read_answers(run_dir)) == 2
    assert stubs["report"] == [run_dir]   # summarised even without damage checks
    assert stubs["damage"] == []


def test_run_with_backend_refuses_questions_from_another_split(project, stubs):
    from bench.backends.dry import OracleBackend

    with pytest.raises(ValueError, match="not in the 'test' split"):
        run.run_with_backend(OracleBackend(), "label", "test", make_questions()[1:3], None)
