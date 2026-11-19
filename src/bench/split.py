"""Freeze the dev / test / gemma_extra split into exams/split.json.

The exam dates for test and gemma_extra come from config/split.toml. Every other exam that
has questions is dev. The frozen file records the question ids of each split, the question
counts per section, the sha256 of questions.jsonl and the config that was used, so a later
change to the questions or the config cannot silently change the split.

The split is written once. freeze_split refuses to overwrite it unless force=True.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

from bench.paths import QUESTIONS, SPLIT, SPLIT_CONFIG
from bench.records import SECTIONS, read_jsonl

NAMED_SPLITS = ("test", "gemma_extra")   # listed in config/split.toml; everything else is dev
DEV = "dev"


def load_split_config(path: Path) -> dict[str, list[str]]:
    """The exam dates of each named split. A date may not be in two splits."""
    with open(path, "rb") as f:
        config = tomllib.load(f)
    named = {name: list(config.get(name, [])) for name in NAMED_SPLITS}
    dates = [d for dates in named.values() for d in dates]
    duplicates = sorted({d for d in dates if dates.count(d) > 1})
    if duplicates:
        raise ValueError(f"{path} lists these exam dates in more than one split: {duplicates}")
    return named


def split_name_of(exam_date: str, config: dict[str, list[str]]) -> str:
    for name in NAMED_SPLITS:
        if exam_date in config.get(name, []):
            return name
    return DEV


def freeze_split(force: bool = False) -> None:
    """Write exams/split.json from the current questions and config/split.toml."""
    _freeze(QUESTIONS, SPLIT_CONFIG, SPLIT, force)


def _freeze(questions_path: Path, config_path: Path, split_path: Path, force: bool) -> None:
    if split_path.exists() and not force:
        raise FileExistsError(f"{split_path} already exists. The split is frozen; "
                              "pass force=True (bench split --force) only to replace it.")
    config = load_split_config(config_path)
    questions = list(read_jsonl(questions_path))
    exam_dates = sorted({q["exam_date"] for q in questions})
    warnings = _warn_about_empty_configured_dates(config, exam_dates)

    splits = {}
    for name in (*NAMED_SPLITS, DEV):
        in_split = [q for q in questions if split_name_of(q["exam_date"], config) == name]
        splits[name] = {
            "exams": sorted({q["exam_date"] for q in in_split}),
            "question_ids": [q["id"] for q in in_split],
            "counts": _section_counts(in_split),
        }

    payload = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions_sha256": _sha256_of(questions_path),
        "config": {"path": str(config_path.name), **config},
        "warnings": warnings,
        "splits": splits,
    }
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, split in splits.items():
        print(f"{name}: {len(split['exams'])} exams, {len(split['question_ids'])} questions "
              f"({split['counts']})")
    print(f"Wrote {split_path}")


def _warn_about_empty_configured_dates(config: dict[str, list[str]], exam_dates: list[str]) -> list[str]:
    warnings = []
    for name in NAMED_SPLITS:
        for exam_date in config[name]:
            if exam_date not in exam_dates:
                message = f"configured {name} exam {exam_date} has no questions"
                warnings.append(message)
                print(f"WARNING: {message}", file=sys.stderr)
    return warnings


def _section_counts(questions: list[dict]) -> dict[str, int]:
    counts = {section: sum(1 for q in questions if q["section"] == section) for section in SECTIONS}
    counts["total"] = len(questions)
    return counts


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
