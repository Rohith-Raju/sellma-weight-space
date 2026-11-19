"""The two record types every part of the bench agrees on, and JSONL helpers.

A Question is one exam question with its official answer.
An Answer is what one model did with one question.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

SECTIONS = ("ORD", "LAS", "MEK")   # LAS = LÄS, written without the diacritic in ids and file names
LETTERS = ("A", "B", "C", "D", "E")


@dataclass
class Question:
    id: str                    # "<exam_date>/p<pass>/<section>/<number>", e.g. "2025-10-19/p3/MEK/21"
    exam_date: str             # "2025-10-19"
    exam_pass: int             # provpass number in the booklet, e.g. 3
    section: str               # one of SECTIONS
    number: int                # question number printed in the booklet, 1-30
    question: str              # ORD: the word or phrase; LAS: the question; MEK: sentence with "_____" gaps
    options: dict[str, str]    # {"A": "...", "B": "...", ...}; ORD and MEK may have 4 or 5 options
    answer: str                # correct letter, from the official answer key
    passage: str | None = None         # LAS only: the full reading text the question is about
    passage_title: str | None = None   # LAS only
    source: dict = field(default_factory=dict)   # {"booklet": path, "page": int, "answer_key": path}

    @staticmethod
    def make_id(exam_date: str, exam_pass: int, section: str, number: int) -> str:
        return f"{exam_date}/p{exam_pass}/{section}/{number}"


@dataclass
class Answer:
    question_id: str
    section: str
    correct_letter: str
    chosen_letter: str               # letter with the highest probability
    is_correct: bool
    letter_probs: dict[str, float]   # raw probability of each option letter as the model's next token
    prob_on_letters: float           # sum of letter_probs; low means the model wanted to say something else
    prob_correct: float              # letter_probs[correct] / prob_on_letters (probability among the options)


def write_jsonl(path: Path, rows: Iterable) -> int:
    """Write dataclasses or dicts as one JSON object per line. Returns the number of rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            data = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_questions(path: Path) -> list[Question]:
    return [Question(**row) for row in read_jsonl(path)]


def load_answers(path: Path) -> list[Answer]:
    return [Answer(**row) for row in read_jsonl(path)]
