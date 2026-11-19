"""Runs one model over one split and writes runs/<UTC time>_<model>_<split>/.

Each run folder holds:

    run.json       what was run: model config, backend description, input checksums,
                   git commit, command, start and finish times. Written first, so a
                   crashed run still says what it was.
    answers.jsonl  one Answer per question, flushed line by line as the run goes.
    damage.json    written by bench.damage (skipped with skip_damage=True)
    summary.json   written by bench.report

The questions are scored from the model's next-token probabilities (see bench.backends),
so the result of a question does not depend on sampling or on the order of the questions.

run_model loads a model, runs it and closes it. run_with_backend does the scoring part for
a backend that is already loaded (and possibly edited in memory), on any list of questions
from one split, and leaves the backend open; bench.experiment uses it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bench import paths
from bench.backends import make_backend
from bench.backends.base import Backend
from bench.models import get_model
from bench.prompts import build_prompt, render_prompt
from bench.records import Answer, Question, load_questions

PROGRESS_EVERY = 50


def run_model(model_name: str, split: str, limit: int | None = None, skip_damage: bool = False) -> Path:
    """Score one model on one split and return the run folder."""
    spec = get_model(model_name)
    # Read the inputs before loading any model, so a missing split fails in a second, not after minutes of loading.
    question_ids, _, frozen_questions_sha256 = read_split(split)
    questions_sha256 = _sha256_file(_require_file(paths.QUESTIONS, "run `uv run bench extract` first"))
    _warn_if_questions_changed(frozen_questions_sha256, questions_sha256)
    questions = select_questions(load_questions(paths.QUESTIONS), question_ids, limit)
    tokenizer_path = str(spec.tokenizer) if spec.tokenizer is not None else None
    _require_tokenizer_for_real_models(spec.backend, tokenizer_path, model_name)

    backend = make_backend(model_name)
    try:
        return run_with_backend(
            backend, model_name, split, questions, tokenizer_path,
            skip_damage=skip_damage,
            extra_info={"model_config": spec.raw, "limit": limit},
            backend_type=spec.backend,
            # `bench run --skip-damage` has always left the summary out; kept that way.
            summarize=not skip_damage,
        )
    finally:
        backend.close()


def run_with_backend(backend: Backend, label: str, split: str, questions: list[Question],
                     tokenizer_path: str | None, skip_damage: bool = False,
                     extra_info: dict[str, Any] | None = None, run_dir: Path | None = None,
                     backend_type: str | None = None, summarize: bool = True) -> Path:
    """Score an already loaded backend on these questions and write a run folder. Does not close the backend.

    This is what the edit experiment (bench.experiment) uses: the model is loaded once, then an
    edit is applied in memory, scored, and removed again, many times over.

    label         written as `model` in run.json (e.g. "qwen-9b-full+edit[2013-10-26/p1/LAS/3]")
    split         the split the questions come from; every question must belong to it
    extra_info    extra keys for run.json, such as the edit that was applied
    run_dir       where to write; a new runs/<UTC time>_<label>_<split>/ folder when None
    backend_type  written as `backend` in run.json; defaults to what backend.describe() says
    summarize     write summary.json and summary.md (bench.report) at the end
    """
    question_ids, split_sha256, frozen_questions_sha256 = read_split(split)
    _require_questions_in_split(questions, question_ids, split)
    questions_sha256 = _sha256_file(_require_file(paths.QUESTIONS, "run `uv run bench extract` first"))

    description = backend.describe()
    backend_type = backend_type or str(description.get("backend"))
    _require_tokenizer_for_real_models(backend_type, tokenizer_path, label)

    run_dir = _prepare_run_dir(run_dir) if run_dir is not None else _new_run_dir(label, split)
    run_info = _start_run_info(label, backend_type, split, description,
                               questions_sha256, split_sha256, frozen_questions_sha256,
                               skip_damage, extra_info or {})
    _write_json(run_dir / "run.json", run_info)

    print(f"Running {label} on {split}: {len(questions)} questions -> {run_dir}", flush=True)
    _answer_all(backend, questions, tokenizer_path, run_dir / "answers.jsonl")

    if not skip_damage:
        _run_damage(backend, run_dir)
    if summarize:
        _summarize(run_dir)

    _finish_run_info(run_dir, run_info, len(questions))
    return run_dir


def read_split(split: str) -> tuple[list[str], str, str | None]:
    """Question ids of one split, the split file's checksum, and the questions checksum recorded when it was frozen."""
    path = _require_file(paths.SPLIT, "run `uv run bench split` first")
    raw = path.read_bytes()
    data = json.loads(raw)
    splits = data.get("splits") if isinstance(data, dict) else None
    if not isinstance(splits, dict) or split not in splits:
        available = ", ".join(sorted(splits)) if isinstance(splits, dict) else "none"
        raise KeyError(f"split {split!r} not found in {path}; available splits: {available}")
    entry = splits[split]
    question_ids = entry.get("question_ids") if isinstance(entry, dict) else None
    if not isinstance(question_ids, list):
        raise ValueError(f"split {split!r} in {path} has no question_ids list")
    return [str(qid) for qid in question_ids], hashlib.sha256(raw).hexdigest(), data.get("questions_sha256")


def select_questions(all_questions: list[Question], question_ids: list[str], limit: int | None) -> list[Question]:
    """The questions of the split, in split order, optionally only the first `limit`."""
    by_id = {question.id: question for question in all_questions}
    missing = [qid for qid in question_ids if qid not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} question ids in the split are not in questions.jsonl, "
                         f"for example {missing[:3]}")
    ids = question_ids if limit is None else question_ids[:limit]
    return [by_id[qid] for qid in ids]


def score_question(backend: Backend, question: Question, tokenizer_path: str | None) -> Answer:
    """Ask the model one question and turn its letter probabilities into an Answer."""
    prompt = render_prompt(build_prompt(question), tokenizer_path)
    letters = sorted(question.options)
    raw_probs = backend.letter_probabilities(prompt, letters, question)
    return score_answer(question, raw_probs)


def score_answer(question: Question, raw_probs: dict[str, float]) -> Answer:
    """Pure scoring: no model involved, so it can be tested directly.

    prob_on_letters is how much of the model's next-token probability landed on the option
    letters at all. A low value means the model wanted to say something else (for example
    a sentence), and the letter ranking is then less meaningful.
    """
    letters = sorted(question.options)
    letter_probs = {letter: float(raw_probs.get(letter, 0.0)) for letter in letters}
    prob_on_letters = sum(letter_probs.values())
    # max() breaks ties by the first letter, so the result is deterministic.
    chosen = max(letters, key=lambda letter: letter_probs[letter])
    prob_correct = letter_probs.get(question.answer, 0.0) / prob_on_letters if prob_on_letters > 0 else 0.0
    # With no probability on any letter, the "chosen" letter is only the tie-break (A), not an
    # answer, so it never counts as correct.
    is_correct = prob_on_letters > 0 and chosen == question.answer
    return Answer(
        question_id=question.id,
        section=question.section,
        correct_letter=question.answer,
        chosen_letter=chosen,
        is_correct=is_correct,
        letter_probs=letter_probs,
        prob_on_letters=prob_on_letters,
        prob_correct=prob_correct,
    )


def _answer_all(backend: Backend, questions: list[Question], tokenizer_path: str | None, answers_path: Path) -> None:
    """Score every question; write answers.jsonl and, next to it, prompts.jsonl with the exact
    prompt text the model saw for each question (chat template included), so any answer can be
    checked against what was actually asked."""
    correct = 0
    prompts_path = answers_path.with_name("prompts.jsonl")
    with open(answers_path, "w", encoding="utf-8") as f, open(prompts_path, "w", encoding="utf-8") as prompts_file:
        for index, question in enumerate(questions, start=1):
            prompt = render_prompt(build_prompt(question), tokenizer_path)
            answer = score_answer(question, backend.letter_probabilities(prompt, sorted(question.options), question))
            prompts_file.write(json.dumps({"question_id": question.id, "prompt": prompt}, ensure_ascii=False) + "\n")
            f.write(json.dumps(asdict(answer), ensure_ascii=False) + "\n")
            f.flush()   # a crash keeps every finished answer
            correct += answer.is_correct
            if index % PROGRESS_EVERY == 0 or index == len(questions):
                print(f"  {index}/{len(questions)} questions, running accuracy {correct / index:.1%}", flush=True)


# Imported when used: these modules are written separately and are not needed for scoring alone.
def _run_damage(backend: Backend, run_dir: Path) -> None:
    damage = importlib.import_module("bench.damage")
    damage.run_damage_checks(backend, run_dir)


def _summarize(run_dir: Path) -> None:
    report = importlib.import_module("bench.report")
    report.summarize_run(run_dir)


REAL_MODEL_BACKENDS = ("hf", "llamacpp")


def _require_tokenizer_for_real_models(backend_type: str, tokenizer_path: str | None, name: str) -> None:
    """A real model must see its own chat format. Without a tokenizer folder the prompt would
    silently fall back to plain text, and the scores would not be comparable with other runs."""
    if backend_type in REAL_MODEL_BACKENDS and tokenizer_path is None:
        raise ValueError(f"{name} uses the {backend_type!r} backend but has no tokenizer folder; "
                         "set `tokenizer` in config/models.toml")


def _require_questions_in_split(questions: list[Question], split_ids: list[str], split: str) -> None:
    known = set(split_ids)
    outside = [question.id for question in questions if question.id not in known]
    if outside:
        raise ValueError(f"{len(outside)} questions are not in the {split!r} split, for example {outside[:3]}")


def _prepare_run_dir(run_dir: Path) -> Path:
    """A run folder chosen by the caller. It may exist, but only empty, so no run is ever mixed with another."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"run folder {run_dir} is not empty")
    return run_dir


def _new_run_dir(model_name: str, split: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = paths.RUNS / f"{stamp}_{model_name}_{split}"
    run_dir = base
    counter = 2
    # Two runs of the same model in the same second must not share a folder.
    while True:
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            run_dir = base.with_name(f"{base.name}_{counter}")
            counter += 1


def _start_run_info(label: str, backend_type: str, split: str, description: dict,
                    questions_sha256: str, split_sha256: str, frozen_questions_sha256: str | None,
                    skip_damage: bool, extra_info: dict[str, Any]) -> dict[str, Any]:
    commit, dirty = _git_state()
    info: dict[str, Any] = {
        "model": label,
        "backend": backend_type,
        "split": split,
        **extra_info,   # `bench run`: model_config and limit; the experiment: the edit that was applied
        "describe": description,
        "questions_sha256": questions_sha256,
        "split_sha256": split_sha256,
        "questions_sha256_at_freeze": frozen_questions_sha256,
        "git_commit": commit,
        "git_dirty": dirty,
        "command": shlex.join(sys.argv),
        "skip_damage": skip_damage,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return info


def _finish_run_info(run_dir: Path, run_info: dict[str, Any], n_questions: int) -> None:
    run_info["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_info["n_questions"] = n_questions
    _write_json(run_dir / "run.json", run_info)


def _warn_if_questions_changed(frozen: str | None, current: str) -> None:
    # The split was frozen against a specific questions.jsonl. If the file has changed since,
    # the split no longer refers to the questions it was made from.
    if frozen is not None and frozen != current:
        print(f"WARNING: exams/questions.jsonl differs from the one the split was frozen with "
              f"({current[:12]} now, {frozen[:12]} at freeze)", flush=True)


def _git_state() -> tuple[str | None, bool | None]:
    commit = _git(["rev-parse", "HEAD"])
    if commit is None:
        return None, None
    status = _git(["status", "--porcelain"])
    return commit.strip(), bool(status and status.strip())


def _git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=paths.ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _sha256_file(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _require_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; {hint}")
    return path


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
