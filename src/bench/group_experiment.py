"""The group edit experiment: one small weight change, trained on many wrong answers at once.

The per-question experiment (bench.experiment) changed the model to fix one question at a time,
and asked whether that helped anywhere else. This experiment asks the broader question:

    If we train one small change to the model on many of its wrong answers together, does it
    then also get right some wrong answers it never saw, without getting right answers wrong?

In plain words, for one model (for example `gemma-4b-full`):

1. split      Start from the model's ordinary full-precision run on the dev split (23 exam
              papers, Phase 1). Put the papers in a fixed but arbitrary order (sorted by the
              sha256 of the exam date). The first third (8 papers) are "hidden": nothing looks at
              them until the final measurement. The next 3 are "validation" papers: training
              watches the score on them to decide when to stop, but never trains on them. The
              rest (12 papers) are the "drill" papers the change is trained on. Whole papers are
              split, not single questions, so a hidden reading question never shares its text
              with a drill question. Each group's questions are then split by whether the
              unedited model got them right: drill_wrong, drill_right, validation_wrong, ...
2. train      Train one low-rank change (a small extra matrix added to a few of the model's
              weight matrices; bench.group_edit) that makes the model answer the drill_wrong
              questions correctly while keeping the drill_right ones right. There is no fixed
              target: training goes on as long as it helps and stops when the score on the
              validation papers (LÄS+MEK accuracy) stops improving ("early stopping"). The change
              kept is the one from the step with the best validation score. Then a random change
              of exactly the same shape and size (per layer) is made: the control. Any change
              moves some answers by chance; the control shows how many.
3. evaluate   Score three versions of the model on the same questions, all in one loaded copy of
              the model so that only the change differs: the unedited model, the model with the
              trained change, and the model with the random change. Question sets: the hidden
              papers (480), the drill papers (720), the validation papers (150), the test split
              (180, with the damage checks) and, for Gemma, gemma_extra (120).
4. quantized  Write the model with the trained change built in to disk, compress it to 4-bit
              exactly like the 4-bit baseline (Q4_K_M), and score it on hidden, test (with damage
              checks) and gemma_extra, against the unedited 4-bit model. The random change is
              compressed too with --with-control-q4.
5. collect    results.json and results.md: the training curve, the numbers below, and a verdict
              in plain words.

How to read the results
-----------------------
* fixed:  of the questions the unedited model got wrong, the share the changed model gets right.
* kept:   of the questions the unedited model got right, the share still right. Every one not
          kept is "broken". Net change = fixed minus broken.
* The drill numbers only say how far training got. They say nothing about Swedish in general:
  the change was trained on exactly those questions. The validation numbers are a little
  flattering too, because the stopping step was chosen by them.
* The key number is on the hidden papers: the share of hidden wrong answers fixed by the trained
  change minus the share fixed by the random change, with a 95% confidence interval (a paired
  bootstrap over questions, bench.stats). Only if the interval lies entirely above zero did the
  training teach the model something that carries over to questions it never saw. If it
  includes zero, the results are no evidence of that, and the report never calls it a gain.
* The test split gets the usual decision rule (bench.report, config/decision.toml): score up
  with the interval above zero, better than the random change, no damage.

At full precision, "wrong" and "right" are the groups in split.json (the Phase 1 run). The
in-memory re-run of the unedited model should agree with them; results.json records how many
questions it disagrees on. At 4-bit, the groups come from the unedited 4-bit model's own answers
on the hidden papers, because compressing changes which questions a model gets wrong.

Resuming. Every step writes its files and is skipped when they already exist and are complete
(a run folder is complete when run.json has `finished_at`; training is complete when
trained/train_result.json exists, which is written last). An unfinished run folder is deleted
and redone. The model is loaded only when there is work left. Every step prints one timestamped
line per action. Training itself cannot resume halfway: an interrupted training starts again.

Folder layout, per model:

    experiments/group/<model>/
      split.json                          step 1: dates, question ids and counts of the six groups
      trained/
        delta.json/.safetensors           the trained change (bench.group_edit.save_delta)
        training_log.jsonl                one line per evaluation during training
        train_result.json                 chosen step, why training stopped, settings
        hidden/ drill/ validation/ test/ gemma_extra/    run folders, full precision (step 3)
        q4_hidden/ q4_test/ q4_gemma_extra/              run folders, 4-bit (step 4)
      control/                            the random change: delta.*, control.json, same run folders
      baseline/                           the unedited model: same full-precision run folders,
                                          q4_hidden/ (and q4_test/, q4_gemma_extra/ only when there
                                          is no Phase 1 4-bit run on those splits)
      comparisons/<name>/                 bench.report comparisons (step 5)
      results.json / results.md           step 5
    work/group/<model>/<condition>/       temporary merged model folder and GGUF (step 4), deleted
                                          after scoring

Dry models (backend random/oracle/broken) have no weights; with a stand-in for bench.group_edit
(see tests/test_group_experiment.py) the whole flow runs without a GPU. Compression then writes
a placeholder file and the "4-bit model" is the dry model's 4-bit entry in config/models.toml.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from bench import paths, report
from bench.backends import make_backend
from bench.backends.base import Backend
from bench.experiment import (
    DEFAULT_QUANT, DRY_BACKENDS, LazyBackend, Measurement, answers_by_id, compact_comparison,
    latest_complete_run, log, pending, q4_model_for, read_json, rel, run_is_complete, sha256_hex,
    split_questions, to_json, write_json_atomic, write_text_atomic,
)
from bench.models import ModelSpec, get_model
from bench.records import LETTERS, SECTIONS, Answer, Question, load_answers
from bench.run import run_with_backend
from bench.stats import bootstrap_mean_difference, wilson_interval

STEPS = ("split", "train", "evaluate", "quantized", "collect", "all")

BASELINE, TRAINED, CONTROL = "baseline", "trained", "control"
CONDITIONS = (BASELINE, TRAINED, CONTROL)
EDITED = (TRAINED, CONTROL)
PARTS = ("drill", "validation", "hidden")                    # the three groups of dev papers
GROUPS = tuple(f"{part}_{result}" for part in PARTS for result in ("wrong", "right"))
HIDDEN_SHARE = 1 / 3          # the first round(n_papers * HIDDEN_SHARE) papers are hidden
VALIDATION_PAPERS = 3         # the next ones are validation papers; the rest are drill papers
CONTROL_SEED = 1
DELTA = "delta"               # save_delta adds .json and .safetensors
CURVE_ROWS_SHOWN = 25         # results.md shows at most this many points of the training curve

# The hidden groups planned for Gemma (from its Phase 1 dev run). A different count is not an
# error, only a sign that the baseline run is not the one the experiment was planned with.
EXPECTED_COUNTS = {"gemma-4b-full": {"hidden_wrong": 117, "hidden_right": 363}}
IN_MEMORY_NOTE = ("`describe` lists the weight files on disk; the change under `experiment.delta` "
                  "was added to the weights in memory for this run")


# ================================================================== paths and small helpers

def group_root() -> Path:
    return paths.ROOT / "experiments" / "group"


def model_dir(model: str) -> Path:
    return group_root() / model


def condition_dir(model: str, condition: str) -> Path:
    return model_dir(model) / condition


def work_dir(model: str, condition: str) -> Path:
    return paths.ROOT / "work" / "group" / model / condition


def split_path(model: str) -> Path:
    return model_dir(model) / "split.json"


def train_result_path(model: str) -> Path:
    return condition_dir(model, TRAINED) / "train_result.json"


def training_log_path(model: str) -> Path:
    return condition_dir(model, TRAINED) / "training_log.jsonl"


def marker_path(model: str, condition: str) -> Path:
    """The file that says a change is saved and complete (written after the change itself)."""
    return train_result_path(model) if condition == TRAINED else condition_dir(model, CONTROL) / "control.json"


def group_edit() -> Any:
    """bench.group_edit, imported when first needed (it needs torch; the tests put a stand-in here)."""
    return importlib.import_module("bench.group_edit")


def is_dry(spec: ModelSpec) -> bool:
    return spec.backend in DRY_BACKENDS


def tokenizer_of(spec: ModelSpec) -> str | None:
    return str(spec.tokenizer) if spec.tokenizer is not None else None


def describe_delta(delta: Any) -> dict:
    """A change without its matrices, for run.json and results."""
    names = ("kind", "layers", "param_names", "rank", "seed", "info")
    return {name: to_json(getattr(delta, name)) for name in names if hasattr(delta, name)}


def weights_of(backend: Backend) -> Any:
    """What bench.group_edit.applied changes: the torch model of an HF backend. A dry backend has
    no weights, so the backend itself is passed (only a test stand-in knows what to do with it)."""
    return getattr(backend, "model", backend)


def run_label(model: str, condition: str) -> str:
    return {BASELINE: model, TRAINED: f"{model}+group[trained]", CONTROL: f"{model}+group[random]"}[condition]


# ================================================================== step 1: split

def split_dates(dates: list[str]) -> dict[str, list[str]]:
    """The fixed rule: sort the exam dates by sha256(date) hex; the first round(n/3) are hidden,
    the next VALIDATION_PAPERS are validation, the rest drill. Each list in date order."""
    ordered = sorted(set(dates), key=sha256_hex)
    n_hidden = round(len(ordered) * HIDDEN_SHARE)
    n_validation = n_hidden + VALIDATION_PAPERS
    return {"hidden": sorted(ordered[:n_hidden]),
            "validation": sorted(ordered[n_hidden:n_validation]),
            "drill": sorted(ordered[n_validation:])}


def make_split(model: str) -> dict:
    """Step 1. Write split.json (once) and return it."""
    path = split_path(model)
    if path.exists():
        data = read_json(path)
        log(model, f"split: done already ({rel(path)}): " + counts_text(data["counts"]))
        return data

    baseline = latest_complete_run(model, "dev")
    answers = {answer.question_id: answer for answer in load_answers(baseline / "answers.jsonl")}
    dev = split_questions("dev")
    missing = [question.id for question in dev if question.id not in answers]
    if missing:
        raise ValueError(f"{baseline.name} has no answer for {len(missing)} dev questions, e.g. {missing[:3]}")

    dates = split_dates([question.exam_date for question in dev])
    part_of = {date: part for part, part_dates in dates.items() for date in part_dates}
    groups: dict[str, list[str]] = {name: [] for name in GROUPS}
    for question in dev:   # split order, so every list is in a fixed order
        result = "right" if answers[question.id].is_correct else "wrong"
        groups[f"{part_of[question.exam_date]}_{result}"].append(question.id)

    sections = {question.id: question.section for question in dev}
    counts = {name: len(ids) for name, ids in groups.items()}
    data = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "baseline_run": rel(baseline),
        "rule": f"exam dates of the dev split sorted by sha256(date) hex; the first round(n/3) are hidden, "
                f"the next {VALIDATION_PAPERS} validation, the rest drill (trained on); right/wrong is the "
                "baseline run's is_correct",
        "dates": dates,
        "counts": counts,
        "counts_by_section": {name: {section: sum(1 for qid in ids if sections[qid] == section)
                                     for section in SECTIONS} for name, ids in groups.items()},
        "question_ids": groups,
    }
    expected = EXPECTED_COUNTS.get(model)
    if expected is not None and any(counts[name] != n for name, n in expected.items()):
        log(model, "WARNING: split counts differ from the planned ones ("
                   + ", ".join(f"{name} {n}" for name, n in expected.items()) + ")")
    write_json_atomic(path, data)
    log(model, f"split: {len(dates['hidden'])} hidden, {len(dates['validation'])} validation and "
               f"{len(dates['drill'])} drill papers from {baseline.name}: {counts_text(counts)} -> {rel(path)}")
    return data


def counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {counts[name]}" for name in GROUPS if name in counts)


# ================================================================== question sets

def has_gemma_extra(model: str) -> bool:
    return model.startswith("gemma")


def questions_of(split: dict, names: list[str]) -> list[Question]:
    """Dev questions in the given groups of split.json, in split order."""
    wanted = {qid for name in names for qid in split["question_ids"][name]}
    return [question for question in split_questions("dev") if question.id in wanted]


def question_sets(model: str, split: dict) -> dict[str, Measurement]:
    """The full-precision question sets of step 3, by name (the name is also the run folder name)."""
    sets = {part: Measurement(part, "dev", questions_of(split, [f"{part}_wrong", f"{part}_right"]))
            for part in ("hidden", "drill", "validation")}
    sets["test"] = Measurement("test", "test", split_questions("test"), damage=True)
    if has_gemma_extra(model):
        sets["gemma_extra"] = Measurement("gemma_extra", "gemma_extra", split_questions("gemma_extra"))
    return sets


Q4_SETS = ("hidden", "test", "gemma_extra")


def q4_question_sets(model: str, split: dict) -> dict[str, Measurement]:
    """The 4-bit question sets of step 4, run folders named q4_<set>."""
    return {name: Measurement(f"q4_{name}", m.split, m.questions, m.damage)
            for name, m in question_sets(model, split).items() if name in Q4_SETS}


# ================================================================== scoring

def score(backend: Backend, label: str, run_dir: Path, measurement: Measurement, tokenizer_path: str | None,
          experiment_info: dict, model: str) -> None:
    """One run folder. An unfinished folder from an interrupted run is removed first."""
    if run_dir.exists():
        log(model, f"removing unfinished run folder {rel(run_dir)}")
        shutil.rmtree(run_dir)
    checks = ", with damage checks" if measurement.damage else ""
    log(model, f"scoring {label} on {measurement.name}: {len(measurement.questions)} questions{checks}")
    run_with_backend(backend, label, measurement.split, measurement.questions, tokenizer_path,
                     skip_damage=not measurement.damage,
                     extra_info={"experiment": {**experiment_info, "question_set": measurement.name}},
                     run_dir=run_dir)
    summary = read_json(run_dir / "summary.json")
    log(model, f"finished {rel(run_dir)}: accuracy {summary['overall']['accuracy']:.1%}")


def measure(get_backend: Callable[[], Backend], folder: Path, label: str, measurements: list[Measurement],
            tokenizer_path: str | None, model: str, info: dict, delta: Any = None) -> None:
    """Produce the missing run folders, with `delta` added to the weights in memory while they run."""
    todo = pending(folder, measurements)
    if not todo:
        log(model, f"{label}: {len(measurements)} runs done already")
        return
    backend = get_backend()
    with delta_applied(backend, delta) as change:
        experiment_info = {"base_model": model, **info}
        if delta is not None:
            experiment_info.update({"delta": describe_delta(delta), "applied_change": to_json(change),
                                    "note": IN_MEMORY_NOTE})
        for measurement in todo:
            score(backend, label, folder / measurement.name, measurement, tokenizer_path, experiment_info, model)


@contextmanager
def delta_applied(backend: Backend, delta: Any) -> Iterator[Any]:
    if delta is None:
        yield None
        return
    with group_edit().applied(weights_of(backend), delta) as change:
        yield change


def full_precision_session(model: str) -> LazyBackend:
    # make_backend is looked up when called (not bound here), so tests can replace it.
    return LazyBackend(model, lambda: make_backend(model), model)


# ================================================================== step 2: train

def train(model: str, session: LazyBackend | None = None, max_steps: int | None = None) -> dict:
    """Step 2. Train on the drill papers, stopping by the validation papers; then make the control."""
    split = make_split(model)
    record = read_json(train_result_path(model))
    if record is not None:
        log(model, f"train: done already ({rel(train_result_path(model))}: chosen step "
                   f"{record['chosen_step']} of {record['steps']})")
        if max_steps is not None and record["settings"].get("max_steps") != max_steps:
            log(model, f"WARNING: the saved training used max_steps={record['settings'].get('max_steps')}, "
                       f"not {max_steps}; delete {rel(condition_dir(model, TRAINED))} and "
                       f"{rel(condition_dir(model, CONTROL))} to train again")
    else:
        record = train_delta(model, split, session, max_steps)
    make_control(model)
    return record


def train_delta(model: str, split: dict, session: LazyBackend | None, max_steps: int | None) -> dict:
    ge = group_edit()
    folder = condition_dir(model, TRAINED)
    folder.mkdir(parents=True, exist_ok=True)
    wrong = questions_of(split, ["drill_wrong"])
    right = questions_of(split, ["drill_right"])
    validation = questions_of(split, ["validation_wrong", "validation_right"])
    settings = {"max_steps": max_steps} if max_steps is not None else {}

    own_session = session is None
    session = session or full_precision_session(model)
    try:
        backend = session.get()
        log(model, f"training on {len(wrong)} drill wrong and {len(right)} drill right questions, "
                   f"stopping by {len(validation)} validation questions (log: {rel(training_log_path(model))})")
        result = ge.train_group_delta(backend, wrong, right, training_log_path(model), validation=validation,
                                      **settings)
    finally:
        if own_session:
            session.close()

    ge.save_delta(result.delta, folder / DELTA)
    training_info = (getattr(result.delta, "info", None) or {}).get("training") or {}
    final = to_json(result.final) if isinstance(result.final, dict) else {}
    record = {
        "model": model,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "steps": to_json(result.steps),
        "chosen_step": final.get("step", training_info.get("best_step")),
        "stopping": {key: to_json(value) for key, value in training_info.items()
                     if any(word in key for word in ("stop", "best", "reason", "patience", "steps_run"))},
        "final": final,
        "log_path": rel(Path(result.log_path)) if result.log_path is not None else None,
        "delta": describe_delta(result.delta),
        "settings": {"max_steps": max_steps, "n_drill_wrong": len(wrong), "n_drill_right": len(right),
                     "n_validation": len(validation), "other": "bench.group_edit.train_group_delta defaults"},
    }
    write_json_atomic(train_result_path(model), record)   # last: marks training as done
    log(model, f"train: chose step {record['chosen_step']} after {result.steps} steps -> {rel(folder)}")
    return record


def make_control(model: str) -> None:
    """The random change of the same per-layer size as the trained one. Made once."""
    if marker_path(model, CONTROL).exists():
        log(model, f"control: done already ({rel(marker_path(model, CONTROL))})")
        return
    ge = group_edit()
    trained = ge.load_delta(condition_dir(model, TRAINED) / DELTA)
    control = ge.random_delta_like(trained, CONTROL_SEED)
    ge.save_delta(control, condition_dir(model, CONTROL) / DELTA)
    write_json_atomic(marker_path(model, CONTROL),
                      {"seed": CONTROL_SEED, "control_for": TRAINED, "delta": describe_delta(control)})
    log(model, f"control: random change with seed {CONTROL_SEED} -> {rel(condition_dir(model, CONTROL))}")


def load_saved_delta(model: str, condition: str) -> Any:
    if not marker_path(model, condition).exists():
        raise FileNotFoundError(f"no {rel(marker_path(model, condition))}; "
                                f"run `bench group train --model {model}` first")
    return group_edit().load_delta(condition_dir(model, condition) / DELTA)


# ================================================================== step 3: evaluate

def evaluate(model: str, session: LazyBackend | None = None, conditions: tuple[str, ...] = CONDITIONS) -> None:
    """Step 3. Score the unedited model and both changes on every question set, in one loaded model."""
    split = make_split(model)
    spec = get_model(model)
    measurements = list(question_sets(model, split).values())
    own_session = session is None
    session = session or full_precision_session(model)
    try:
        for condition in conditions:
            folder = condition_dir(model, condition)
            if not pending(folder, measurements):
                log(model, f"evaluate {condition}: {len(measurements)} runs done already")
                continue
            delta = None if condition == BASELINE else load_saved_delta(model, condition)
            measure(session.get, folder, run_label(model, condition), measurements, tokenizer_of(spec), model,
                    info={"condition": condition, "precision": "full"}, delta=delta)
    finally:
        if own_session:
            session.close()
    log(model, "evaluate: finished")


# ================================================================== step 4: 4-bit

def quantized(model: str, with_control: bool = False) -> None:
    """Step 4. 4-bit baseline on the hidden papers, then each change merged, compressed and scored."""
    split = make_split(model)
    spec = get_model(model)
    q4_name = q4_model_for(model)
    q4_spec = get_model(q4_name)
    q4_tokenizer = tokenizer_of(q4_spec)
    sets = q4_question_sets(model, split)

    # The unedited 4-bit model: Phase 1 runs where they exist, our own runs for the rest.
    own = [m for name, m in sets.items() if q4_phase1_run(model, name) is None]
    baseline = LazyBackend(model, lambda: make_backend(q4_name), f"{q4_name} (unedited 4-bit)")
    try:
        measure(baseline.get, condition_dir(model, BASELINE), q4_name, own, q4_tokenizer, model,
                info={"condition": BASELINE, "precision": "q4"})
    finally:
        baseline.close()

    for condition in (EDITED if with_control else (TRAINED,)):
        quantize_one(model, spec, q4_spec, q4_tokenizer, condition, list(sets.values()))
    log(model, "quantized: finished")


def q4_phase1_run(model: str, set_name: str) -> Path | None:
    """The Phase 1 run of the 4-bit model on a whole split (test, gemma_extra). None for the
    hidden papers, which are not a whole split, and when there is no such run."""
    if set_name not in ("test", "gemma_extra"):
        return None
    try:
        return latest_complete_run(q4_model_for(model), set_name, need_damage=(set_name == "test"))
    except (FileNotFoundError, ValueError):
        return None


def quantize_one(model: str, spec: ModelSpec, q4_spec: ModelSpec, q4_tokenizer: str | None, condition: str,
                 measurements: list[Measurement]) -> None:
    folder = condition_dir(model, condition)
    label = run_label(q4_spec.name, condition)
    if not pending(folder, measurements):
        log(model, f"{label}: {len(measurements)} runs done already")
        return
    delta = load_saved_delta(model, condition)
    work = work_dir(model, condition)
    quant = str(q4_spec.setting("compression", DEFAULT_QUANT))
    gguf = merged_gguf(model, spec, delta, work, quant)
    backend = LazyBackend(model, lambda: open_gguf(gguf, q4_spec), f"4-bit changed model {rel(gguf)}")
    try:
        # The change is in the file, so nothing is applied in memory here.
        measure(backend.get, folder, label, measurements, q4_tokenizer, model,
                info={"condition": condition, "precision": "q4", "delta_in_file": describe_delta(delta),
                      "gguf": rel(gguf), "quant": quant})
    finally:
        backend.close()
    shutil.rmtree(work)   # several GB; it can be rebuilt from the saved change
    log(model, f"deleted {rel(work)}")


def merged_gguf(model: str, spec: ModelSpec, delta: Any, work: Path, quant: str) -> Path:
    """Write the changed model to disk, compress it like the 4-bit baseline, delete the full-size copy."""
    gguf = work / f"merged-{quant}.gguf"
    if gguf_done(gguf):
        log(model, f"reusing finished {rel(gguf)}")
        return gguf
    if spec.path is None and not is_dry(spec):
        raise ValueError(f"model {model!r} has no path in config/models.toml")
    hf_dir = work / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)   # left over from an interrupted run
    work.mkdir(parents=True, exist_ok=True)
    log(model, f"saving the changed model to {rel(hf_dir)}")
    hf_dir = Path(group_edit().save_merged_model(spec.path, delta, hf_dir))
    log(model, f"compressing to {quant}: {rel(gguf)}")
    compress(spec, hf_dir, gguf, quant)
    shutil.rmtree(hf_dir)
    return gguf


def gguf_done(gguf: Path) -> bool:
    # bench.compress writes the file under its final name only when complete, and a .json record next to it.
    return gguf.exists() and gguf.with_name(gguf.name + ".json").exists()


def compress(spec: ModelSpec, hf_dir: Path, gguf: Path, quant: str) -> Path:
    if is_dry(spec):
        gguf.write_text("dry placeholder, not a model\n", encoding="utf-8")
        write_json_atomic(gguf.with_name(gguf.name + ".json"), {"source": str(hf_dir), "quant": quant})
        return gguf
    from bench.compress import compress_model

    return compress_model(hf_dir, gguf, quant=quant)


def open_gguf(gguf: Path, q4_spec: ModelSpec) -> Backend:
    if is_dry(q4_spec):
        return make_backend(q4_spec.name)
    from bench.backends.llamacpp import LlamaCppBackend

    bin_dir = q4_spec.setting("llama_cpp_bin")
    # Same tokenizer as the 4-bit baseline, so changed and unedited 4-bit models read the same token ids.
    # No port is given: the backend asks the operating system for a free one, so it never clashes
    # with a llama-server of another job on the same machine.
    return LlamaCppBackend(gguf_path=gguf, llama_bin_dir=Path(bin_dir).expanduser() if bin_dir else None,
                           tokenizer_path=q4_spec.tokenizer)


# ================================================================== step 5: collect

def collect(model: str) -> Path:
    """Step 5. results.json and results.md from whatever runs exist. Cheap, so it always runs."""
    split = make_split(model)
    decision = report.load_decision()
    training = read_json(train_result_path(model))
    results = {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": {key: split[key] for key in ("baseline_run", "dates", "counts", "counts_by_section")},
        "training": training,
        "training_curve": training_curve(training_log_path(model), training),
        "confidence": decision["confidence"],
        "full": precision_results(model, split, "full", decision),
        "q4": precision_results(model, split, "q4", decision),
    }
    results["verdict"] = verdict_paragraph(results)
    out = model_dir(model) / "results.json"
    write_json_atomic(out, results)
    write_text_atomic(model_dir(model) / "results.md", render_results_md(results))
    log(model, f"collect: -> {rel(out)} and results.md")
    return out


def training_curve(log_path: Path, training: dict | None) -> dict | None:
    """The training log reduced to one point per evaluation: step, validation score, drill numbers.

    The validation score is the log's validation accuracy field (the one with "headline" in its
    name when there are several), found by name so a renamed field still shows up."""
    if not log_path.exists():
        return None
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None
    key = validation_key(rows)
    points = [{"step": row.get("step"), "validation": row.get(key) if key else None,
               "drill_wrong_fixed": row.get("drill_wrong_fixed"), "drill_right_kept": row.get("drill_right_kept")}
              for row in rows]
    scored = [p for p in points if isinstance(p["validation"], (int, float))]
    best = max(scored, key=lambda p: p["validation"]) if scored else None
    return {"validation_key": key, "points": points,
            "chosen_step": (training or {}).get("chosen_step"),
            "best_logged": best,
            "start": points[0], "last": points[-1]}


def validation_key(rows: list[dict]) -> str | None:
    numeric = [k for k, v in rows[-1].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    candidates = [k for k in numeric if "valid" in k.lower() and "acc" in k.lower()]
    for preferred in ("headline", ""):
        for k in candidates:
            if preferred in k.lower():
                return k
    return None


def run_name(precision: str, set_name: str) -> str:
    return set_name if precision == "full" else f"q4_{set_name}"


def baseline_run(model: str, precision: str, set_name: str) -> Path:
    if precision == "q4":
        phase1 = q4_phase1_run(model, set_name)
        if phase1 is not None:
            return phase1
    return condition_dir(model, BASELINE) / run_name(precision, set_name)


def complete_answers(run_dir: Path) -> dict[str, Answer] | None:
    return answers_by_id(run_dir) or None


def precision_results(model: str, split: dict, precision: str, decision: dict) -> dict | None:
    """All numbers of one precision (full or q4). None when neither change has been run at it."""
    parts = PARTS if precision == "full" else ("hidden",)
    sets = [*parts, "test"] if precision == "full" else ["hidden", "test"]
    if has_gemma_extra(model):
        sets.append("gemma_extra")

    def run(condition: str, set_name: str) -> Path:
        return condition_dir(model, condition) / run_name(precision, set_name)

    after = {c: {s: complete_answers(run(c, s)) for s in sets} for c in EDITED}
    if not any(answers for runs in after.values() for answers in runs.values()):
        return None

    groups, groups_from, baseline_check = reference_groups(model, split, precision)
    out: dict[str, Any] = {"groups_from": groups_from, "baseline_check": baseline_check}
    for condition in EDITED:
        row: dict[str, Any] = {}
        for part in parts:
            answers = after[condition].get(part)
            if f"{part}_wrong" in groups and answers is not None:
                row[part] = fixed_and_kept(answers, groups[f"{part}_wrong"], groups[f"{part}_right"], decision)
                row[f"{part}_by_section"] = by_section(answers, groups[f"{part}_wrong"], groups[f"{part}_right"])
        for s in sets:
            control = run(CONTROL, s) if condition == TRAINED else None
            row[f"{s}_vs_baseline"] = compare_pair(model, f"{precision}_{condition}_vs_baseline_{s}", run(condition, s),
                                                   baseline_run(model, precision, s), control)
        out[condition] = row

    versus: dict[str, Any] = {}
    for part in parts:
        trained, control = after[TRAINED].get(part), after[CONTROL].get(part)
        if f"{part}_wrong" in groups and trained is not None and control is not None:
            wrong, right = groups[f"{part}_wrong"], groups[f"{part}_right"]
            versus[f"{part}_wrong_fixed"] = paired(trained, control, wrong, decision)
            versus[f"{part}_right_kept"] = paired(trained, control, right, decision)
            versus[f"{part}_net"] = paired(trained, control, wrong + right, decision)
    for s in sets:
        versus[s] = compare_pair(model, f"{precision}_trained_vs_control_{s}", run(TRAINED, s), run(CONTROL, s), None)
    out["trained_vs_control"] = versus
    return out


def reference_groups(model: str, split: dict, precision: str) -> tuple[dict, str | None, dict]:
    """Which questions count as wrong and right before the change, where that comes from, and
    how well the unedited re-run agrees. Full precision: split.json. 4-bit: the unedited 4-bit
    model on the hidden papers."""
    if precision == "full":
        groups = {name: list(split["question_ids"][name]) for name in GROUPS}
        check = {}
        for part in PARTS:
            rerun = complete_answers(condition_dir(model, BASELINE) / part)
            if rerun is None:
                check[part] = None
                continue
            disagree = [qid for qid in groups[f"{part}_wrong"] if rerun[qid].is_correct]
            disagree += [qid for qid in groups[f"{part}_right"] if not rerun[qid].is_correct]
            check[part] = {"run": rel(condition_dir(model, BASELINE) / part), "disagreements": len(disagree),
                           "question_ids": disagree}
        return groups, split["baseline_run"], check

    run_dir = baseline_run(model, "q4", "hidden")
    answers = complete_answers(run_dir)
    if answers is None:
        return {}, None, {}
    hidden_ids = split["question_ids"]["hidden_wrong"] + split["question_ids"]["hidden_right"]
    return {"hidden_wrong": [qid for qid in hidden_ids if not answers[qid].is_correct],
            "hidden_right": [qid for qid in hidden_ids if answers[qid].is_correct]}, rel(run_dir), {}


def fixed_and_kept(answers: dict[str, Answer], wrong: list[str], right: list[str], decision: dict) -> dict:
    """Fixed and kept counts and shares, and the net change against the unedited model."""
    fixed = sum(answers[qid].is_correct for qid in wrong)
    kept = sum(answers[qid].is_correct for qid in right)
    net = None
    if wrong or right:
        after = [float(answers[qid].is_correct) for qid in wrong + right]
        before = [0.0] * len(wrong) + [1.0] * len(right)
        change, low, high = bootstrap_mean_difference(after, before, confidence=decision["confidence"],
                                                      samples=decision["bootstrap_samples"])
        net = {"accuracy_change": change, "ci": [low, high]}
    return {
        "n_wrong": len(wrong), "fixed": fixed, "fixed_share": share(fixed, len(wrong)),
        "fixed_ci": list(wilson_interval(fixed, len(wrong))) if wrong else None,
        "n_right": len(right), "kept": kept, "kept_share": share(kept, len(right)),
        "kept_ci": list(wilson_interval(kept, len(right))) if right else None,
        "broken": len(right) - kept,
        "net_questions": fixed - (len(right) - kept),
        "net": net,
    }


def by_section(answers: dict[str, Answer], wrong: list[str], right: list[str]) -> dict:
    rows = {}
    for section in SECTIONS:
        w = [qid for qid in wrong if answers[qid].section == section]
        r = [qid for qid in right if answers[qid].section == section]
        fixed = sum(answers[qid].is_correct for qid in w)
        kept = sum(answers[qid].is_correct for qid in r)
        rows[section] = {"n_wrong": len(w), "fixed": fixed, "fixed_share": share(fixed, len(w)),
                         "n_right": len(r), "kept": kept, "kept_share": share(kept, len(r)),
                         "broken": len(r) - kept}
    return rows


def paired(first: dict[str, Answer], second: dict[str, Answer], ids: list[str], decision: dict) -> dict | None:
    """Share correct under `first` minus under `second`, on the same questions, with a paired bootstrap CI."""
    if not ids:
        return None
    a = [float(first[qid].is_correct) for qid in ids]
    b = [float(second[qid].is_correct) for qid in ids]
    change, low, high = bootstrap_mean_difference(a, b, confidence=decision["confidence"],
                                                  samples=decision["bootstrap_samples"])
    return {"n": len(ids), "first": int(sum(a)), "second": int(sum(b)), "change": change, "ci": [low, high]}


def share(count: int, n: int) -> float | None:
    return count / n if n else None


def compare_pair(model: str, name: str, edited: Path, baseline: Path, control: Path | None) -> dict | None:
    """bench.report.compare_runs on two (or three) run folders; None while a run is missing."""
    if not run_is_complete(edited) or not run_is_complete(baseline):
        return None
    if control is not None and not run_is_complete(control):
        control = None
    out_root = model_dir(model) / "comparisons" / name
    with contextlib.redirect_stdout(io.StringIO()):   # compare_runs prints the whole report; it is in comparison.md
        result = report.compare_runs(edited, baseline, control=control, out_root=out_root)
    return compact_comparison(result, out_root / f"{result['edited']}__vs__{result['baseline']}")


# ================================================================== the verdict in plain words

def above_zero(ci: list[float] | None) -> bool:
    return ci is not None and ci[0] > 0


def below_zero(ci: list[float] | None) -> bool:
    return ci is not None and ci[1] < 0


def pp(value: float) -> str:
    return f"{100 * value:+.1f} percentage points"


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def ci_text(ci: list[float]) -> str:
    return f"95% CI {100 * ci[0]:+.1f} to {100 * ci[1]:+.1f}"


def verdict_paragraph(results: dict) -> str:
    """Fixed rules, so the same numbers always give the same words. Nothing is called a gain or an
    improvement unless its confidence interval lies entirely above zero."""
    training = results.get("training")
    if training is None:
        return "The change has not been trained yet, so there is nothing to report."
    sentences: list[str] = []
    drill = ((results.get("full") or {}).get(TRAINED) or {}).get("drill")
    chosen = f"step {training['chosen_step']} of {training['steps']}, chosen by the validation papers"
    if drill is not None:
        sentences.append(
            f"Training kept the change from {chosen}. It fixes {pct(drill['fixed_share'])} of the "
            f"{drill['n_wrong']} drill questions the model got wrong and keeps {pct(drill['kept_share'])} of the "
            f"{drill['n_right']} it got right. These are the questions it was trained on, so this only shows how "
            "far training got.")
    else:
        sentences.append(f"Training kept the change from {chosen}; the drill papers have not been scored yet.")

    for precision, name in (("full", "At full precision"), ("q4", "After compression to 4-bit")):
        numbers = results.get(precision)
        if numbers is None:
            if precision == "q4":
                sentences.append("The 4-bit version has not been measured yet.")
            continue
        sentences.append("\n\n" + " ".join(precision_sentences(numbers, name)))
    return " ".join(sentences).replace(" \n\n", "\n\n")


def precision_sentences(numbers: dict, name: str) -> list[str]:
    out: list[str] = []
    hidden = (numbers.get(TRAINED) or {}).get("hidden")
    control_hidden = (numbers.get(CONTROL) or {}).get("hidden")
    key = numbers.get("trained_vs_control", {}).get("hidden_wrong_fixed")
    if hidden is None:
        return [f"{name}, the hidden papers have not been scored yet."]

    if key is None or control_hidden is None:
        out.append(f"{name}, the trained change fixes {hidden['fixed']} of {hidden['n_wrong']} wrong answers on the "
                   f"hidden papers ({pct(hidden['fixed_share'])}). There is no random-change run to compare with, "
                   "so it is not known whether that is more than any change of this size would do.")
    else:
        compared = (f"{name}, on the hidden papers (never used in training or for stopping) the trained change "
                    f"fixes {pct(hidden['fixed_share'])} of the {hidden['n_wrong']} wrong answers, the random "
                    f"change {pct(control_hidden['fixed_share'])}: a difference of {pp(key['change'])} "
                    f"({ci_text(key['ci'])}).")
        if above_zero(key["ci"]):
            out.append(compared + " The interval lies above zero, so the change does generalise: it fixes more "
                                  "unseen wrong answers than a random change of the same size.")
        elif below_zero(key["ci"]):
            out.append(compared + " The interval lies below zero: the trained change fixes fewer unseen wrong "
                                  "answers than a random change, so it does not generalise.")
        else:
            out.append(compared + " The interval includes zero, so this is no evidence that the change "
                                  "generalises: it fixes no more unseen wrong answers than a random change "
                                  "might by chance.")

    net = hidden["net"]
    cost = (f"It breaks {hidden['broken']} of the {hidden['n_right']} hidden questions the model got right "
            f"(keeps {pct(hidden['kept_share'])}); the net change on the hidden papers is "
            f"{hidden['net_questions']:+d} questions, {pp(net['accuracy_change'])} accuracy ({ci_text(net['ci'])})")
    if above_zero(net["ci"]):
        out.append(cost + ", a real net gain over the unedited model.")
    elif below_zero(net["ci"]):
        out.append(cost + ", a real net loss against the unedited model.")
    else:
        out.append(cost + "; the interval includes zero, so the net change cannot be told apart from no change.")

    test = (numbers.get(TRAINED) or {}).get("test_vs_baseline")
    if test is not None:
        headline = test["headline"]
        interval = f"95% CI {headline['metric_ci'][0]:+.4f} to {headline['metric_ci'][1]:+.4f}"
        if test["verdict"] == "IMPROVEMENT":
            out.append(f"On the test split the decision rule says IMPROVEMENT: the headline score (LÄS+MEK, mean "
                       f"probability on the correct answer) changes by {headline['metric_change']:+.4f} "
                       f"({interval}), it beats the random change and there is no damage.")
        else:
            failed = [rule for rule, entry in test["rules"].items() if not entry["passed"]]
            out.append(f"On the test split the decision rule says NO IMPROVEMENT (headline change "
                       f"{headline['metric_change']:+.4f}, {interval}; rules not passed: {', '.join(failed)}).")
    return out


# ================================================================== results.md

def render_results_md(results: dict) -> str:
    model = results["model"]
    split = results["split"]
    dates = split["dates"]
    lines = [
        f"# Group edit: {model}",
        "",
        "One small low-rank change, trained on many of the model's wrong answers at once (the drill papers), "
        "stopped when the score on the validation papers stopped improving, then tested on wrong answers it "
        "never saw (the hidden papers), against a random change of the same size. See "
        "`src/bench/group_experiment.py` for the method.",
        "",
        "## Verdict",
        "",
        results["verdict"],
        "",
        "## The split",
        "",
        f"From `{split['baseline_run']}`. Hidden papers: {', '.join(dates['hidden'])}. "
        f"Validation papers: {', '.join(dates['validation'])}. Drill papers: the other {len(dates['drill'])}.",
        "",
        "| Group | Questions | LÄS | MEK | ORD |",
        "|---|---|---|---|---|",
    ]
    for name in GROUPS:
        by = split["counts_by_section"][name]
        lines.append(f"| {name} | {split['counts'][name]} | {by['LAS']} | {by['MEK']} | {by['ORD']} |")

    lines += ["", "## Training", ""] + training_md(results)

    for precision, title in (("full", "Full precision"), ("q4", "4-bit (Q4_K_M)")):
        numbers = results.get(precision)
        lines += ["", f"## {title}", ""]
        if numbers is None:
            lines.append("Not measured yet.")
            continue
        lines += precision_md(numbers, precision)
    lines.append("")
    return "\n".join(lines)


def training_md(results: dict) -> list[str]:
    training, curve = results["training"], results["training_curve"]
    if training is None:
        return ["Not trained yet."]
    lines = [f"* Chosen step: {training['chosen_step']} of {training['steps']} steps run.",
             f"* Why it stopped: {training['stopping'] or 'not recorded'}",
             f"* Log: `{training['log_path']}`"]
    drill = ((results.get("full") or {}).get(TRAINED) or {}).get("drill")
    if drill is not None:
        lines.append(f"* At the chosen step (scored by the bench): drill wrong fixed {pct(drill['fixed_share'])} "
                     f"({drill['fixed']}/{drill['n_wrong']}), drill right kept {pct(drill['kept_share'])} "
                     f"({drill['kept']}/{drill['n_right']}).")
    if curve is None:
        return lines
    key = curve["validation_key"] or "(no validation field found in the log)"
    lines += ["", f"Training curve (from the log; validation = `{key}`). The chosen step is marked.", "",
              "| Step | Validation | Drill wrong fixed | Drill right kept |", "|---|---|---|---|"]
    for point in curve_rows(curve):
        mark = " (chosen)" if point["step"] == curve["chosen_step"] else ""
        validation = point["validation"]
        lines.append(f"| {point['step']}{mark} | {pct(validation) if isinstance(validation, (int, float)) else 'n/a'} | "
                     f"{pct(point['drill_wrong_fixed'])} | {pct(point['drill_right_kept'])} |")
    return lines


def curve_rows(curve: dict) -> list[dict]:
    """At most CURVE_ROWS_SHOWN evenly spaced points, always with the first, the chosen and the last."""
    points = curve["points"]
    if len(points) <= CURVE_ROWS_SHOWN:
        return points
    every = -(-len(points) // CURVE_ROWS_SHOWN)
    keep = {0, len(points) - 1} | set(range(0, len(points), every))
    keep |= {i for i, p in enumerate(points) if p["step"] == curve["chosen_step"]}
    return [points[i] for i in sorted(keep)]


def precision_md(numbers: dict, precision: str) -> list[str]:
    parts = PARTS if precision == "full" else ("hidden",)
    lines = [f"Wrong and right before the change: `{numbers['groups_from']}`."]
    for part, entry in (numbers.get("baseline_check") or {}).items():
        if entry is not None:
            lines.append(f"The unedited re-run on {part} disagrees with that on {entry['disagreements']} questions.")

    lines += ["", "### Fixed, kept and broken", "",
              "| Change | Papers | Wrong fixed | Right kept | Right broken | Net (accuracy, CI) |",
              "|---|---|---|---|---|---|"]
    for condition in EDITED:
        for part in parts:
            entry = (numbers.get(condition) or {}).get(part)
            if entry is None:
                continue
            net = entry["net"]
            lines.append(f"| {condition} | {part} | {pct(entry['fixed_share'])} ({entry['fixed']}/{entry['n_wrong']}) | "
                         f"{pct(entry['kept_share'])} ({entry['kept']}/{entry['n_right']}) | {entry['broken']} | "
                         f"{entry['net_questions']:+d} questions, {100 * net['accuracy_change']:+.1f} pp "
                         f"({ci_text(net['ci'])}) |")

    versus = numbers.get("trained_vs_control", {})
    lines += ["", "### Trained minus random change", "",
              "The key number is the bold row: does the trained change fix more unseen wrong answers than a "
              "random change of the same size? Counts are questions answered correctly.", "",
              "| Questions | n | Trained | Random | Difference (CI) |", "|---|---|---|---|---|"]
    labels = {f"{part}_{kind}": f"{part.capitalize()} {text}" for part in ("hidden", "validation", "drill")
              if part in parts
              for kind, text in (("wrong_fixed", "wrong: fixed"), ("right_kept", "right: kept"), ("net", "all"))}
    labels["hidden_wrong_fixed"] = "**Hidden wrong: fixed**"
    shown = False
    for key in labels:
        entry = versus.get(key)
        if entry is not None:
            shown = True
            lines.append(f"| {labels[key]} | {entry['n']} | {entry['first']} | {entry['second']} | "
                         f"{100 * entry['change']:+.1f} pp ({ci_text(entry['ci'])}) |")
    if not shown:
        lines.append("| (no random-change run at this precision) | | | | |")

    sets = [s for s in ("test", "gemma_extra", "hidden", "validation", "drill")
            if any((numbers.get(c) or {}).get(f"{s}_vs_baseline") for c in EDITED) or versus.get(s)]
    lines += ["", "### Against the unedited model (bench.report)", "",
              "Only the test split gives a verdict that counts; the others are indicative, and on sets without "
              "damage checks the damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on "
              "the correct answer. The decision rule does not apply to trained vs random, so no verdict is shown "
              "there.", "",
              "| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | "
              "Right→wrong | Largest letter shift | Perplexity | Loop rate |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for s in sets:
        pairs = [(f"{c} vs unedited", (numbers.get(c) or {}).get(f"{s}_vs_baseline")) for c in EDITED]
        for label, comparison in pairs:
            if comparison is not None:
                lines.append(f"| {s} | {label} | {comparison_cells(comparison)} |")
        if versus.get(s) is not None:
            lines.append(f"| {s} | trained vs random | {comparison_cells(versus[s], with_verdict=False)} |")

    lines += ["", "### By section (hidden papers)", "",
              "| Change | Section | Wrong fixed | Right kept | Broken |", "|---|---|---|---|---|"]
    for condition in EDITED:
        for section, entry in ((numbers.get(condition) or {}).get("hidden_by_section") or {}).items():
            lines.append(f"| {condition} | {section} | {pct(entry['fixed_share'])} ({entry['fixed']}/{entry['n_wrong']}) | "
                         f"{pct(entry['kept_share'])} ({entry['kept']}/{entry['n_right']}) | {entry['broken']} |")
    test_sections = [(c, (numbers.get(c) or {}).get("test_vs_baseline")) for c in EDITED]
    if any(comparison for _, comparison in test_sections):
        lines += ["", "### By section (test split, against the unedited model)", "",
                  "| Change | Section | n | Accuracy change (pp, CI) | Metric change (CI) |", "|---|---|---|---|---|"]
        for condition, comparison in test_sections:
            for section, group in (comparison or {}).get("sections", {}).items():
                lines.append(f"| {condition} | {section} | {group['n']} | "
                             f"{report.fmt_ci(group['accuracy_change'], group['accuracy_ci'], scale=100)} | "
                             f"{report.fmt_ci(group['metric_change'], group['metric_ci'], decimals=4)} |")

    lines += ["", "### Chosen letters", "",
              "Change in the share of answers choosing each letter, against the unedited model (percentage "
              "points). A change that only pushes the model toward one letter moves answers without learning "
              "anything.", "",
              "| Set | Change | " + " | ".join(LETTERS) + " |", "|---|---|" + "---|" * len(LETTERS)]
    for s in sets:
        for condition in EDITED:
            comparison = (numbers.get(condition) or {}).get(f"{s}_vs_baseline")
            if comparison is not None:
                shift = comparison["letter_shift"]["overall"]
                lines.append(f"| {s} | {condition} | " + " | ".join(f"{100 * shift[x]:+.1f}" for x in LETTERS) + " |")
    return lines


def comparison_cells(comparison: dict, with_verdict: bool = True) -> str:
    headline, overall, damage = comparison["headline"], comparison["overall"], comparison["damage"]
    verdict = comparison["verdict"] + (" (indicative)" if comparison["indicative_only"] else "")
    if not with_verdict:
        verdict = "—"
    if damage is not None:
        perplexity = f"{damage['perplexity']['relative_change']:+.2%}"
        loops = f"{damage['loop_rate_percent']['change_points']:+.1f} points"
    else:
        perplexity = loops = "n/a"
    return (f"{verdict} | {report.fmt_ci(headline['metric_change'], headline['metric_ci'], decimals=4)} | "
            f"{report.fmt_ci(overall['accuracy_change'], overall['accuracy_ci'], scale=100)} | "
            f"{overall['wrong_to_right']} | {overall['right_to_wrong']} | "
            f"{100 * comparison['letter_shift']['largest_overall']:.1f} pp | {perplexity} | {loops}")


# ================================================================== all steps

def run_all(model: str, with_control_q4: bool = False, max_steps: int | None = None) -> None:
    """Steps 1 to 5. The unedited model is scored before training, so nothing training does to the
    loaded model can touch the baseline. The full-precision model is loaded once and closed before
    llama.cpp needs the GPU."""
    make_split(model)
    session = full_precision_session(model)
    try:
        evaluate(model, session, conditions=(BASELINE,))
        train(model, session, max_steps=max_steps)
        evaluate(model, session, conditions=EDITED)
    finally:
        session.close()
    quantized(model, with_control=with_control_q4)
    collect(model)
    log(model, "all steps finished")


def run_step(step: str, model: str, with_control_q4: bool = False, max_steps: int | None = None) -> None:
    """Entry point for `bench group STEP --model MODEL`."""
    if step == "split":
        make_split(model)
    elif step == "train":
        train(model, max_steps=max_steps)
    elif step == "evaluate":
        evaluate(model)
    elif step == "quantized":
        quantized(model, with_control=with_control_q4)
    elif step == "collect":
        collect(model)
    elif step == "all":
        run_all(model, with_control_q4=with_control_q4, max_steps=max_steps)
    else:
        raise ValueError(f"unknown step {step!r}; steps: {', '.join(STEPS)}")
