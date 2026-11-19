"""Tests for the frozen split. Every test uses temporary files; the real exams/split.json is never written."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bench.split import DEV, _freeze, load_split_config, split_name_of

CONFIG = """
test = ["2025-10-19", "2026-10-18"]
gemma_extra = ["2024-10-20"]
"""


def write_questions(path: Path) -> Path:
    rows = []
    for exam_date, section, number in [
        ("2025-10-19", "ORD", 1), ("2025-10-19", "LAS", 11), ("2025-10-19", "MEK", 21),
        ("2024-10-20", "ORD", 1), ("2024-10-20", "MEK", 22),
        ("2023-10-22", "ORD", 2),
    ]:
        rows.append({"id": f"{exam_date}/p3/{section}/{number}", "exam_date": exam_date,
                     "section": section, "number": number})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def files(tmp_path):
    config = tmp_path / "split.toml"
    config.write_text(CONFIG, encoding="utf-8")
    questions = write_questions(tmp_path / "questions.jsonl")
    return config, questions, tmp_path / "split.json"


def test_test_and_gemma_extra_exams_are_listed_and_the_rest_is_dev(files):
    config, questions, split_path = files
    _freeze(questions, config, split_path, force=False)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    splits = payload["splits"]
    assert splits["test"]["exams"] == ["2025-10-19"]
    assert splits["gemma_extra"]["exams"] == ["2024-10-20"]
    assert splits["dev"]["exams"] == ["2023-10-22"]
    assert splits["test"]["question_ids"] == [
        "2025-10-19/p3/ORD/1", "2025-10-19/p3/LAS/11", "2025-10-19/p3/MEK/21"]
    assert splits["test"]["counts"] == {"ORD": 1, "LAS": 1, "MEK": 1, "total": 3}


def test_the_split_records_the_questions_hash_the_config_and_a_time(files):
    config, questions, split_path = files
    _freeze(questions, config, split_path, force=False)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert payload["questions_sha256"] == hashlib.sha256(questions.read_bytes()).hexdigest()
    assert payload["config"]["test"] == ["2025-10-19", "2026-10-18"]
    assert payload["created"]


def test_a_configured_exam_without_questions_is_warned_about_loudly(files, capsys):
    config, questions, split_path = files
    _freeze(questions, config, split_path, force=False)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    assert "configured test exam 2026-10-18 has no questions" in payload["warnings"]
    assert "WARNING: configured test exam 2026-10-18 has no questions" in capsys.readouterr().err


def test_an_existing_split_is_not_overwritten_without_force(files):
    config, questions, split_path = files
    _freeze(questions, config, split_path, force=False)
    before = split_path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        _freeze(questions, config, split_path, force=False)
    assert split_path.read_text(encoding="utf-8") == before


def test_force_replaces_an_existing_split(files):
    config, questions, split_path = files
    split_path.write_text("{}", encoding="utf-8")
    _freeze(questions, config, split_path, force=True)
    assert "splits" in json.loads(split_path.read_text(encoding="utf-8"))


def test_an_exam_in_two_named_splits_is_refused(tmp_path):
    config = tmp_path / "split.toml"
    config.write_text('test = ["2025-10-19"]\ngemma_extra = ["2025-10-19"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="more than one split"):
        load_split_config(config)


def test_split_name_of_an_exam():
    config = {"test": ["2025-10-19"], "gemma_extra": ["2024-10-20"]}
    assert split_name_of("2025-10-19", config) == "test"
    assert split_name_of("2024-10-20", config) == "gemma_extra"
    assert split_name_of("2019-04-06", config) == DEV
