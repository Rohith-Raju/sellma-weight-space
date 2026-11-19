"""The edit experiment: fix one wrong answer by changing the model's
weights, and measure whether the model got better at Swedish in general or only on that question.

In plain words, for one model (for example `qwen-9b-full`):

1. select-failures  Take the model's full-precision run on the dev split (Phase 1) and pick
                    questions it got wrong: 4 LÄS, 4 MEK, 2 ORD. Not picked by hand: inside a
                    section, the wrong answers are sorted by the sha256 of their question id and
                    the first ones are taken.
2. dev-subset       A fixed set of 300 dev questions (100 per section) for a second, larger
                    measurement. It leaves out the failure cases of both models, so both models
                    are measured on the same set. Chosen the same way: sorted by sha256 of the id.
3. edits            Load the model once. Score the unedited model on the failure-case questions,
                    the test split and the dev subset (the baseline for everything below). Then,
                    for each failure case: find the smallest rank-1 weight change that makes the
                    model answer that question correctly (bench.edit). If one exists, apply it in
                    memory and score the target question, the test split and the dev subset; then
                    remove it and do the same with a random change of the same size (the control).
4. quantized        For each edit that fixed its question: write the edited model to disk,
                    compress it to 4-bit exactly like the 4-bit baseline (Q4_K_M), and score the
                    target question and the test split again. The 4-bit baseline on test comes
                    from Phase 1. Random controls are compressed only with --with-control-q4.
5. combined         (Phase 4, optional) All of the model's targeted edits at once, against a
                    control made of all their random controls at once. Measured like step 3.
6. collect          One row per edit and per control in results.jsonl: was the target fixed
                    (full precision and 4-bit), the change on test and on the dev subset against
                    the right baseline with paired 95% intervals (bench.report), the decision-rule
                    verdict, damage, and how the chosen-letter shares moved.

Damage checks (perplexity, loop rate) measure the whole model, not one set of questions, so they
run once per edit: with the test-split run, because that is the run the verdict is read from.
The target-question and dev-subset runs skip them.

Resuming. Every step writes its files and is skipped when they already exist and are complete
(a run folder is complete when run.json has `finished_at`). An unfinished run folder is deleted
and redone. The model is loaded only when there is work left, so re-running a finished step
takes seconds. Every step prints one timestamped line per action.

Folder layout, per model:

    experiments/dev_subset.json                 the shared dev subset (both models)
    experiments/<model>/
      failure_cases.jsonl / .md                 step 1 (the .md gets English explanations later)
      baseline/
        full_targets/ full_test/ full_dev_subset/   unedited model, same in-memory backend (step 3)
        q4_targets/                                 unedited 4-bit model on the failure cases (step 4)
      edits/<question id with / as _>/
        search.json                             the ladder search: status, every step tried
        targeted_edit.json/.safetensors         the edit (bench.edit.save_edit)
        control_edit.json/.safetensors          its random control
        targeted_applied_change.json            how much of the edit survived rounding (also control_)
        targeted_target/ targeted_test/ targeted_dev_subset/      run folders (step 3)
        control_target/  control_test/  control_dev_subset/
        targeted_q4_target/ targeted_q4_test/   run folders (step 4; control_q4_* with --with-control-q4)
        comparisons/                            bench.report comparisons (step 6)
      combined/                                 step 5, same names (targeted_targets/, ...)
      results.jsonl                             step 6
    work/<model>/                               temporary edited model folders and GGUF files (step 4)

Dry models. When the model is a dry stand-in (backend random/oracle/broken) there are no weights
to edit; DryTools takes the place of bench.edit, bench.compress and llama.cpp, so the whole flow
can be checked on a laptop and in the tests.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import shutil
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from bench import paths, report
from bench.backends import make_backend
from bench.backends.base import Backend
from bench.models import ModelSpec, get_model, load_model_specs
from bench.records import SECTIONS, Answer, Question, load_answers, load_questions, read_jsonl
from bench.run import read_split, run_with_backend, select_questions

EXPERIMENT_MODELS = ("qwen-9b-full", "gemma-4b-full")   # they share one dev subset
FAILURES_PER_SECTION = {"LAS": 4, "MEK": 4, "ORD": 2}
DEV_SUBSET_PER_SECTION = 100
DRY_BACKENDS = ("random", "oracle", "broken")
DEFAULT_QUANT = "Q4_K_M"

TARGETED = "targeted"
CONTROL = "control"
TARGETED_EDIT = "targeted_edit"   # bench.edit.save_edit adds .json and .safetensors
CONTROL_EDIT = "control_edit"
EXPLANATION_PLACEHOLDER = "(to be written)"
IN_MEMORY_NOTE = ("`describe` lists the weight files on disk; the edits under `edits` were applied "
                  "to the model in memory for this run")
STEPS = ("select-failures", "dev-subset", "edits", "quantized", "combined", "collect", "all")


# ================================================================== paths and small helpers

def experiments_root() -> Path:
    return paths.ROOT / "experiments"


def model_dir(model: str) -> Path:
    return experiments_root() / model


def baseline_dir(model: str) -> Path:
    return model_dir(model) / "baseline"


def edit_dir(model: str, question_id: str) -> Path:
    return model_dir(model) / "edits" / case_slug(question_id)


def combined_dir(model: str) -> Path:
    return model_dir(model) / "combined"


def work_dir(model: str) -> Path:
    return paths.ROOT / "work" / model


def case_slug(question_id: str) -> str:
    """'2013-10-26/p1/LAS/3' -> '2013-10-26_p1_LAS_3' (usable as a folder name)."""
    return question_id.replace("/", "_")


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(text: str) -> int:
    """A seed derived from a text, the same on every machine and every run."""
    return int(sha256_hex(text)[:8], 16)


def control_seed(question_id: str) -> int:
    return stable_seed(f"control|{question_id}")


def log(model: str, message: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"{stamp} [{model}] {message}", flush=True)


def rel(path: Path) -> str:
    """A path relative to the project folder when possible, for files that should stay portable."""
    try:
        return str(Path(path).relative_to(paths.ROOT))
    except ValueError:
        return str(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write to a temporary name and rename, so a crash never leaves a half-written file that looks done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    write_text_atomic(path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def to_json(value: Any) -> Any:
    """Plain JSON values from what bench.edit returns (dataclasses, numpy or torch scalars, ...).
    Vectors are not written out, only their shape."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: to_json(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None:
        return to_json(value.item()) if len(shape) == 0 else f"<array of shape {list(shape)}>"
    return str(value)


EDIT_FIELDS = ("kind", "question_id", "layer", "param_name", "alpha", "alpha_fraction", "seed")


def describe_edit(edit: Any) -> dict:
    """An edit without its vectors, for run.json and results."""
    return {name: to_json(getattr(edit, name)) for name in EDIT_FIELDS if hasattr(edit, name)}


def questions_by_id() -> dict[str, Question]:
    return {question.id: question for question in load_questions(paths.QUESTIONS)}


def split_questions(split: str) -> list[Question]:
    question_ids, _, _ = read_split(split)
    return select_questions(load_questions(paths.QUESTIONS), question_ids, None)


# ================================================================== run folders

def run_is_complete(run_dir: Path, need_damage: bool = False) -> bool:
    """A run folder is complete when run.json says it finished (written last) and summary.json exists."""
    info = read_json(run_dir / "run.json")
    if not isinstance(info, dict) or not info.get("finished_at"):
        return False
    if not (run_dir / "summary.json").exists():
        return False
    return not need_damage or (run_dir / "damage.json").exists()


def latest_complete_run(model: str, split: str, need_damage: bool = False) -> Path:
    """The newest finished Phase 1 run of this model on this whole split (runs with --limit do not count)."""
    found: list[tuple[str, str, Path]] = []
    if paths.RUNS.exists():
        for run_dir in paths.RUNS.iterdir():
            info = read_json(run_dir / "run.json") if run_dir.is_dir() else None
            if not isinstance(info, dict):
                continue
            if (info.get("model") == model and info.get("split") == split and info.get("limit") is None
                    and run_is_complete(run_dir, need_damage)):
                found.append((str(info.get("started_at", "")), run_dir.name, run_dir))
    if not found:
        raise FileNotFoundError(f"no complete {split} run of {model} in {paths.RUNS}; "
                                f"run `uv run bench run {model} --split {split}` first")
    return max(found)[2]


def answers_by_id(run_dir: Path) -> dict[str, Answer]:
    if not run_is_complete(run_dir):
        return {}
    return {answer.question_id: answer for answer in load_answers(run_dir / "answers.jsonl")}


# ================================================================== step 1: failure cases

def failure_cases_path(model: str) -> Path:
    return model_dir(model) / "failure_cases.jsonl"


def pick_failure_cases(answers: list[Answer], per_section: dict[str, int] = FAILURES_PER_SECTION) -> list[Answer]:
    """The fixed rule: per section, the wrong answers sorted by sha256(question id), first k of them."""
    picked: list[Answer] = []
    for section, count in per_section.items():
        wrong = [answer for answer in answers if answer.section == section and not answer.is_correct]
        wrong.sort(key=lambda answer: sha256_hex(answer.question_id))
        picked += wrong[:count]
    return picked


def select_failures(model: str) -> list[dict]:
    """Step 1. Write failure_cases.jsonl and failure_cases.md; return the cases."""
    out = failure_cases_path(model)
    if out.exists():
        log(model, f"select-failures: done already ({rel(out)})")
        return list(read_jsonl(out))

    baseline = latest_complete_run(model, "dev")
    questions = questions_by_id()
    cases = pick_failure_cases(load_answers(baseline / "answers.jsonl"))
    for section, wanted in FAILURES_PER_SECTION.items():
        got = sum(1 for answer in cases if answer.section == section)
        if got < wanted:
            log(model, f"WARNING: only {got} wrong {section} answers in {baseline.name}, wanted {wanted}")

    rows = [failure_row(answer, questions[answer.question_id], baseline) for answer in cases]
    markdown = model_dir(model) / "failure_cases.md"
    if not markdown.exists():   # never overwrite: explanations are added to it by hand or by a later step
        write_text_atomic(markdown, render_failure_cases_md(model, rows, baseline))
    write_jsonl_atomic(out, rows)   # last, because its existence marks the step as done
    log(model, f"select-failures: {len(rows)} cases from {baseline.name} -> {rel(out)}")
    return rows


def failure_row(answer: Answer, question: Question, baseline: Path) -> dict:
    return {
        "question_id": answer.question_id,
        "section": answer.section,
        "correct_letter": answer.correct_letter,
        "chosen_letter": answer.chosen_letter,
        "prob_correct": answer.prob_correct,
        "letter_probs": answer.letter_probs,
        "baseline_run": rel(baseline),
        "question": asdict(question),
    }


def render_failure_cases_md(model: str, rows: list[dict], baseline: Path) -> str:
    lines = [
        f"# Failure cases: {model}",
        "",
        f"Questions the unedited model gets wrong in `{rel(baseline)}` (dev split, full precision), "
        "picked by a fixed rule: inside each section the wrong answers are sorted by "
        "sha256 of the question id and the first ones are taken (4 LÄS, 4 MEK, 2 ORD).",
        "",
        "| # | Question | Section | Correct | Model chose | P(correct) | English explanation |",
        "|---|---|---|---|---|---|---|",
    ]
    for number, row in enumerate(rows, start=1):
        lines.append(f"| {number} | `{row['question_id']}` | {row['section']} | {row['correct_letter']} | "
                     f"{row['chosen_letter']} | {row['prob_correct']:.3f} | {EXPLANATION_PLACEHOLDER} |")
    lines += ["", "## The questions (Swedish, as the model saw them)", ""]
    for number, row in enumerate(rows, start=1):
        question = row["question"]
        lines += [f"### {number}. `{row['question_id']}`", ""]
        if question.get("passage"):
            lines += ["<details><summary>Reading text: " + (question.get("passage_title") or "untitled")
                      + "</summary>", "", question["passage"], "", "</details>", ""]
        lines += [question["question"], ""]
        for letter, option in sorted(question["options"].items()):
            marks = []
            if letter == row["correct_letter"]:
                marks.append("correct")
            if letter == row["chosen_letter"]:
                marks.append("model's choice")
            lines.append(f"* {letter}. {option}" + (f"  ({', '.join(marks)})" if marks else ""))
        lines.append("")
    return "\n".join(lines)


# ================================================================== step 2: dev subset

def experiment_group(model: str) -> tuple[str, ...]:
    """The models that share a dev subset: the two experiment models, or a model on its own."""
    return EXPERIMENT_MODELS if model in EXPERIMENT_MODELS else (model,)


def dev_subset_path(group: tuple[str, ...]) -> Path:
    if tuple(group) == tuple(EXPERIMENT_MODELS):
        return experiments_root() / "dev_subset.json"
    return experiments_root() / f"dev_subset_{'+'.join(group)}.json"


def pick_dev_subset(questions: list[Question], excluded: set[str], per_section: int) -> list[Question]:
    """The fixed rule: per section, questions not excluded, sorted by sha256(question id), first n."""
    chosen: list[Question] = []
    for section in SECTIONS:
        pool = [q for q in questions if q.section == section and q.id not in excluded]
        pool.sort(key=lambda question: sha256_hex(question.id))
        chosen += pool[:per_section]
    return chosen


def dev_subset(model: str, per_section: int = DEV_SUBSET_PER_SECTION) -> list[Question]:
    """Step 2. Write the shared dev subset once; return its questions."""
    group = experiment_group(model)
    path = dev_subset_path(group)
    if path.exists():
        data = read_json(path)
        if model not in data["models"]:
            raise ValueError(f"{path} was made for {data['models']}, not for {model}")
        log(model, f"dev-subset: done already ({rel(path)})")
        by_id = questions_by_id()
        return [by_id[qid] for qid in data["question_ids"]]

    excluded: set[str] = set()
    for member in group:   # needs the failure cases of every model in the group
        excluded |= {row["question_id"] for row in select_failures(member)}
    chosen = pick_dev_subset(split_questions("dev"), excluded, per_section)
    counts = {section: sum(1 for q in chosen if q.section == section) for section in SECTIONS}
    for section, count in counts.items():
        if count < per_section:
            log(model, f"WARNING: dev subset has only {count} {section} questions, wanted {per_section}")
    write_json_atomic(path, {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rule": "per section: dev questions minus every failure case of these models, "
                "sorted by sha256(question id), first per_section",
        "models": list(group),
        "per_section": per_section,
        "counts": counts,
        "excluded_question_ids": sorted(excluded),
        "question_ids": [q.id for q in chosen],
    })
    log(model, f"dev-subset: {len(chosen)} questions -> {rel(path)}")
    return chosen


# ================================================================== inputs shared by steps 3 to 6

@dataclass
class Inputs:
    model: str
    spec: ModelSpec
    tokenizer_path: str | None
    cases: list[dict]              # failure_cases.jsonl rows
    targets: list[Question]        # the failure-case questions, same order
    test: list[Question]
    dev_subset: list[Question]


def load_inputs(model: str) -> Inputs:
    spec = get_model(model)
    cases = select_failures(model)
    subset = dev_subset(model)
    by_id = questions_by_id()
    return Inputs(
        model=model,
        spec=spec,
        tokenizer_path=str(spec.tokenizer) if spec.tokenizer is not None else None,
        cases=cases,
        targets=[by_id[case["question_id"]] for case in cases],
        test=split_questions("test"),
        dev_subset=subset,
    )


@dataclass
class Measurement:
    """One run folder to produce: which questions, recorded under which split, with or without damage checks."""
    name: str
    split: str
    questions: list[Question]
    damage: bool = False


def edit_measurements(variant: str, targets: list[Question], inputs: Inputs, target_name: str = "target",
                      ) -> list[Measurement]:
    """Target question(s), test split, dev subset. Damage checks only on the test run: they measure
    the whole model, so once per edit is enough."""
    return [
        Measurement(f"{variant}_{target_name}", "dev", targets),
        Measurement(f"{variant}_test", "test", inputs.test, damage=True),
        Measurement(f"{variant}_dev_subset", "dev", inputs.dev_subset),
    ]


def pending(folder: Path, measurements: list[Measurement]) -> list[Measurement]:
    return [m for m in measurements if not run_is_complete(folder / m.name, m.damage)]


class LazyBackend:
    """A backend loaded the first time it is needed. A finished step never loads the model."""

    def __init__(self, model: str, factory: Callable[[], Backend], what: str):
        self.model = model
        self.factory = factory
        self.what = what
        self._backend: Backend | None = None
        self._reference_norms: dict[int, float] = {}

    def get(self) -> Backend:
        if self._backend is None:
            log(self.model, f"loading {self.what} (can take minutes)")
            self._backend = self.factory()
        return self._backend

    def reference_norm(self, tools: Any, layer: int) -> float:
        """The spectral norm of the edited matrix, computed once per model and layer (the ladder's unit)."""
        if layer not in self._reference_norms:
            self._reference_norms[layer] = float(tools.reference_norm(self.get(), layer))
        return self._reference_norms[layer]

    def close(self) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None


def measure(get_backend: Callable[[], Backend], tools: Any, folder: Path, label: str, variant: str,
            edits: list, measurements: list[Measurement], tokenizer_path: str | None, model: str,
            info: dict | None = None) -> None:
    """Produce the missing run folders of `measurements`, with `edits` applied in memory while they run."""
    todo = pending(folder, measurements)
    if not todo:
        log(model, f"{label}: {len(measurements)} runs done already")
        return
    backend = get_backend()
    with applied(tools, backend, edits) as changes:
        experiment_info = {"base_model": model, "variant": variant, **(info or {})}
        if edits:
            experiment_info.update({"edits": [describe_edit(e) for e in edits],
                                    "applied_change": to_json(changes), "note": IN_MEMORY_NOTE})
            write_json_atomic(folder / f"{variant}_applied_change.json", to_json(changes))
        for measurement in todo:
            score(backend, label, folder / measurement.name, measurement, tokenizer_path, experiment_info, model)


@contextmanager
def applied(tools: Any, backend: Backend, edits: list) -> Iterator[list[dict]]:
    if not edits:
        yield []
        return
    with tools.applied(backend, edits) as changes:
        yield changes


def score(backend: Backend, label: str, run_dir: Path, measurement: Measurement, tokenizer_path: str | None,
          experiment_info: dict, model: str) -> None:
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


# ================================================================== step 3: one edit per failure case

def run_edits(model: str, tools: Any = None, session: LazyBackend | None = None) -> None:
    """Step 3. Baseline runs, then for each failure case: search, edited runs, control runs."""
    inputs = load_inputs(model)
    tools = tools or tools_for(model)
    own_session = session is None
    session = session or full_precision_session(model)
    try:
        measure(session.get, tools, baseline_dir(model), model, "full", [],
                [Measurement("full_targets", "dev", inputs.targets),
                 Measurement("full_test", "test", inputs.test, damage=True),
                 Measurement("full_dev_subset", "dev", inputs.dev_subset)],
                inputs.tokenizer_path, model)
        for question in inputs.targets:
            edit_one_case(session, tools, inputs, question)
    finally:
        if own_session:
            session.close()
    log(model, "edits: finished")


def full_precision_session(model: str) -> LazyBackend:
    # make_backend is looked up when called (not bound here), so tests can replace it.
    return LazyBackend(model, lambda: make_backend(model), f"{model}")


def edit_one_case(session: LazyBackend, tools: Any, inputs: Inputs, question: Question) -> None:
    model, qid = inputs.model, question.id
    folder = edit_dir(model, qid)
    search = targeted_search(session, tools, inputs, question)
    if not search["fixed"]:
        return
    edit = tools.load_edit(folder / TARGETED_EDIT)
    measure(session.get, tools, folder, f"{model}+edit[{qid}]", TARGETED, [edit],
            edit_measurements(TARGETED, [question], inputs), inputs.tokenizer_path, model)
    control = control_edit(tools, folder, edit, qid)
    measure(session.get, tools, folder, f"{model}+random[{qid}]", CONTROL, [control],
            edit_measurements(CONTROL, [question], inputs), inputs.tokenizer_path, model)


def targeted_search(session: LazyBackend, tools: Any, inputs: Inputs, question: Question) -> dict:
    """Find (or read back) the smallest ladder edit that fixes this question. Writes search.json."""
    model, qid = inputs.model, question.id
    folder = edit_dir(model, qid)
    path = folder / "search.json"
    if path.exists():
        record = read_json(path)
        log(model, f"edit search for {qid}: done already ({record['status']})")
        return record

    layer = edit_layer(inputs.spec)
    reference_norm = session.reference_norm(tools, layer)
    log(model, f"searching the edit for {qid} at layer {layer}")
    search = tools.search(session.get(), question, layer, reference_norm)
    folder.mkdir(parents=True, exist_ok=True)
    if search.fixed:
        tools.save_edit(search.edit, folder / TARGETED_EDIT)
    record = {
        "question_id": qid,
        "section": question.section,
        "fixed": bool(search.fixed),
        "status": getattr(search, "status", "") or ("fixed" if search.fixed else "could_not_fix"),
        "reference_norm": reference_norm,
        "edit": describe_edit(search.edit) if search.fixed else None,
        "tried": to_json(search.tried),
        "baseline": to_json(getattr(search, "baseline", {})),
    }
    write_json_atomic(path, record)   # last: marks the search as done
    if search.fixed:
        log(model, f"{qid}: fixed at alpha fraction {search.edit.alpha_fraction:g} "
                   f"after {len(search.tried)} ladder steps")
    else:
        log(model, f"{qid}: not fixed ({record['status']}) after {len(search.tried)} ladder steps")
    return record


def edit_layer(spec: ModelSpec) -> int:
    layer = spec.setting("edit_layer")
    if layer is None:
        if spec.backend in DRY_BACKENDS:
            return 0
        raise ValueError(f"model {spec.name!r} has no edit_layer in config/models.toml")
    return int(layer)


def control_edit(tools: Any, folder: Path, edit: Any, question_id: str) -> Any:
    """The random control of an edit: same layer and size, seed derived from the question id. Saved once."""
    path = folder / CONTROL_EDIT
    if tools.edit_saved(path):
        return tools.load_edit(path)
    control = tools.random_like(edit, control_seed(question_id))
    tools.save_edit(control, path)
    return control


# ================================================================== step 4: 4-bit

def q4_model_for(model: str) -> str:
    """The 4-bit baseline entry of a full model: the models.toml entry whose `parent` is this model."""
    children = [spec for spec in load_model_specs().values() if spec.setting("parent") == model]
    if len(children) > 1:
        children = [spec for spec in children if spec.setting("compression") == DEFAULT_QUANT]
    if len(children) != 1:
        found = ", ".join(spec.name for spec in children) or "none"
        raise ValueError(f"expected one 4-bit entry with parent = {model!r} in config/models.toml, found {found}")
    return children[0].name


def run_quantized(model: str, tools: Any = None, with_control: bool = False) -> None:
    """Step 4. Save, compress and score every fixed edit at 4-bit (and its control with with_control)."""
    inputs = load_inputs(model)
    tools = tools or tools_for(model)
    q4_name = q4_model_for(model)
    q4_spec = get_model(q4_name)
    q4_tokenizer = str(q4_spec.tokenizer) if q4_spec.tokenizer is not None else None
    latest_complete_run(q4_name, "test", need_damage=True)   # the Phase 1 4-bit baseline must exist

    baseline = LazyBackend(model, lambda: make_backend(q4_name), f"{q4_name} (unedited 4-bit)")
    try:
        measure(baseline.get, tools, baseline_dir(model), q4_name, "q4", [],
                [Measurement("q4_targets", "dev", inputs.targets)], q4_tokenizer, model)
    finally:
        baseline.close()

    variants = [TARGETED, CONTROL] if with_control else [TARGETED]
    for question in inputs.targets:
        search = read_json(edit_dir(model, question.id) / "search.json")
        if search is None:
            raise RuntimeError(f"no edit search for {question.id}; run `bench experiment edits --model {model}` first")
        if not search["fixed"]:
            continue
        for variant in variants:
            quantize_one(tools, inputs, q4_spec, q4_tokenizer, question, variant)
    log(model, "quantized: finished")


def quantize_one(tools: Any, inputs: Inputs, q4_spec: ModelSpec, q4_tokenizer: str | None,
                 question: Question, variant: str) -> None:
    model, qid = inputs.model, question.id
    folder = edit_dir(model, qid)
    label = f"{q4_spec.name}+{'edit' if variant == TARGETED else 'random'}[{qid}]"
    measurements = [Measurement(f"{variant}_q4_target", "dev", [question]),
                    Measurement(f"{variant}_q4_test", "test", inputs.test, damage=True)]
    if not pending(folder, measurements):
        log(model, f"{label}: {len(measurements)} runs done already")
        return
    edit_path = folder / (TARGETED_EDIT if variant == TARGETED else CONTROL_EDIT)
    if not tools.edit_saved(edit_path):
        raise RuntimeError(f"{edit_path} is missing; run `bench experiment edits --model {model}` first")
    edit = tools.load_edit(edit_path)

    work = work_dir(model) / f"{case_slug(qid)}_{variant}"
    gguf = edited_gguf(tools, inputs, q4_spec, edit, work)
    backend = LazyBackend(model, lambda: tools.open_gguf(gguf, q4_spec), f"4-bit edited model {rel(gguf)}")
    try:
        # The edit is in the file, so nothing is applied in memory here.
        measure(backend.get, tools, folder, label, f"{variant}_q4", [], measurements, q4_tokenizer, model,
                info={"edits_in_file": [describe_edit(edit)], "gguf": rel(gguf),
                      "quant": str(q4_spec.setting("compression", DEFAULT_QUANT))})
    finally:
        backend.close()
    shutil.rmtree(work)   # the GGUF is several GB; it can be rebuilt from the saved edit
    log(model, f"deleted {rel(work)}")


def edited_gguf(tools: Any, inputs: Inputs, q4_spec: ModelSpec, edit: Any, work: Path) -> Path:
    """Write the edited model to disk, compress it like the 4-bit baseline, delete the full-size copy."""
    model = inputs.model
    quant = str(q4_spec.setting("compression", DEFAULT_QUANT))
    gguf = work / f"edited-{quant}.gguf"
    if tools.gguf_done(gguf):
        log(model, f"reusing finished {rel(gguf)}")
        return gguf
    if inputs.spec.path is None and inputs.spec.backend not in DRY_BACKENDS:
        raise ValueError(f"model {model!r} has no path in config/models.toml")
    hf_dir = work / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)   # left over from an interrupted run
    work.mkdir(parents=True, exist_ok=True)
    log(model, f"saving the edited model to {rel(hf_dir)}")
    hf_dir = Path(tools.save_edited_model(inputs.spec.path, edit, hf_dir))
    log(model, f"compressing to {quant}: {rel(gguf)}")
    tools.compress(hf_dir, gguf, quant)
    shutil.rmtree(hf_dir)
    return gguf


# ================================================================== step 5: combined edit (Phase 4)

def combined_edit(model: str, tools: Any = None, session: LazyBackend | None = None) -> None:
    """Step 5. All fixed targeted edits applied together, against all their random controls together."""
    inputs = load_inputs(model)
    tools = tools or tools_for(model)
    fixed: list[Question] = []
    for question in inputs.targets:
        search = read_json(edit_dir(model, question.id) / "search.json")
        if search is None:
            raise RuntimeError(f"no edit search for {question.id}; run `bench experiment edits --model {model}` first")
        if search["fixed"]:
            fixed.append(question)
    if len(fixed) < 2:
        log(model, f"combined: only {len(fixed)} fixed edit(s), nothing to combine")
        return

    edits = [tools.load_edit(edit_dir(model, q.id) / TARGETED_EDIT) for q in fixed]
    controls = [control_edit(tools, edit_dir(model, q.id), edit, q.id) for q, edit in zip(fixed, edits)]
    folder = combined_dir(model)
    folder.mkdir(parents=True, exist_ok=True)
    write_json_atomic(folder / "combined.json", {
        "question_ids": [q.id for q in fixed],
        "edits": [describe_edit(e) for e in edits],
        "controls": [describe_edit(c) for c in controls],
    })

    own_session = session is None
    session = session or full_precision_session(model)
    try:
        n = len(edits)
        measure(session.get, tools, folder, f"{model}+edits[combined {n}]", TARGETED, edits,
                edit_measurements(TARGETED, fixed, inputs, target_name="targets"), inputs.tokenizer_path, model)
        measure(session.get, tools, folder, f"{model}+random[combined {n}]", CONTROL, controls,
                edit_measurements(CONTROL, fixed, inputs, target_name="targets"), inputs.tokenizer_path, model)
    finally:
        if own_session:
            session.close()
    log(model, "combined: finished")


# ================================================================== step 6: collect

@dataclass
class Baselines:
    before_full: dict[str, Answer]     # unedited full precision, failure-case questions
    before_q4: dict[str, Answer]       # unedited 4-bit, failure-case questions
    full_test: Path
    full_dev_subset: Path
    q4_test: Path | None               # Phase 1 run of the 4-bit model on test


def collect(model: str) -> Path:
    """Step 6. Write results.jsonl: one row per edit and per control. Cheap, so it always runs."""
    inputs = load_inputs(model)
    base = baseline_dir(model)
    baselines = Baselines(
        before_full=answers_by_id(base / "full_targets"),
        before_q4=answers_by_id(base / "q4_targets"),
        full_test=base / "full_test",
        full_dev_subset=base / "full_dev_subset",
        q4_test=phase1_q4_test(model),
    )
    rows: list[dict] = []
    for question in inputs.targets:
        rows += case_rows(inputs, question, baselines)
    rows += combined_rows(inputs, baselines)
    out = model_dir(model) / "results.jsonl"
    write_jsonl_atomic(out, rows)
    log(model, f"collect: {len(rows)} rows -> {rel(out)}")
    return out


def phase1_q4_test(model: str) -> Path | None:
    try:
        return latest_complete_run(q4_model_for(model), "test", need_damage=True)
    except (FileNotFoundError, ValueError):
        return None


def case_rows(inputs: Inputs, question: Question, baselines: Baselines) -> list[dict]:
    qid = question.id
    folder = edit_dir(inputs.model, qid)
    search = read_json(folder / "search.json")
    common = {
        "model": inputs.model,
        "question_id": qid,
        "section": question.section,
        "status": search["status"] if search else "not_run",
        "fixed": search["fixed"] if search else None,
        "alpha_fraction": (search.get("edit") or {}).get("alpha_fraction") if search else None,
        "alpha": (search.get("edit") or {}).get("alpha") if search else None,
        "tried_steps": len(search["tried"]) if search else None,
        "tried": search["tried"] if search else None,
    }
    if not search or not search["fixed"]:
        return [{**common, "kind": TARGETED,
                 "target": {"full": {"before": answer_brief(baselines.before_full.get(qid)), "after": None}}}]

    rows = []
    for variant in (TARGETED, CONTROL):
        control = CONTROL if variant == TARGETED else None
        edit_file = folder / f"{TARGETED_EDIT if variant == TARGETED else CONTROL_EDIT}.json"
        q4_after = answer_brief(answers_by_id(folder / f"{variant}_q4_target").get(qid))
        q4_test = compare(folder, f"{variant}_q4_test", baselines.q4_test, control and f"{control}_q4_test")
        row = {
            **common,
            "kind": variant,
            "edit": read_json(edit_file),
            "applied_change": read_json(folder / f"{variant}_applied_change.json"),
            "target": {
                "full": {"before": answer_brief(baselines.before_full.get(qid)),
                         "after": answer_brief(answers_by_id(folder / f"{variant}_target").get(qid))},
                "q4": {"before": answer_brief(baselines.before_q4.get(qid)), "after": q4_after},
            },
            "test": compare(folder, f"{variant}_test", baselines.full_test, control and f"{control}_test"),
            "dev_subset": compare(folder, f"{variant}_dev_subset", baselines.full_dev_subset,
                                  control and f"{control}_dev_subset"),
            "q4_test": q4_test,
        }
        if variant == TARGETED:
            row["survives_quantization"] = survives_quantization(q4_after, q4_test)
        rows.append(row)
    return rows


def combined_rows(inputs: Inputs, baselines: Baselines) -> list[dict]:
    folder = combined_dir(inputs.model)
    description = read_json(folder / "combined.json")
    if description is None:
        return []
    rows = []
    for variant, kind in ((TARGETED, "combined"), (CONTROL, "combined_control")):
        control = CONTROL if variant == TARGETED else None
        after = answers_by_id(folder / f"{variant}_targets")
        rows.append({
            "model": inputs.model,
            "kind": kind,
            "question_ids": description["question_ids"],
            "edits": description["edits" if variant == TARGETED else "controls"],
            "applied_change": read_json(folder / f"{variant}_applied_change.json"),
            "target": {"full": {qid: {"before": answer_brief(baselines.before_full.get(qid)),
                                      "after": answer_brief(after.get(qid))}
                                for qid in description["question_ids"]}},
            "test": compare(folder, f"{variant}_test", baselines.full_test, control and f"{control}_test"),
            "dev_subset": compare(folder, f"{variant}_dev_subset", baselines.full_dev_subset,
                                  control and f"{control}_dev_subset"),
        })
    return rows


def answer_brief(answer: Answer | None) -> dict | None:
    if answer is None:
        return None
    return {"chosen_letter": answer.chosen_letter, "is_correct": answer.is_correct,
            "prob_correct": answer.prob_correct, "prob_on_letters": answer.prob_on_letters}


def survives_quantization(q4_target_after: dict | None, q4_test: dict | None) -> bool | None:
    """Survival rule: at 4-bit the target stays fixed and rule 1 still holds against the 4-bit baseline."""
    if q4_target_after is None or q4_test is None:
        return None
    return bool(q4_target_after["is_correct"] and q4_test["rules"]["1_metric_improves"]["passed"])


def compare(folder: Path, edited_name: str, baseline: Path | None, control_name: str | None) -> dict | None:
    """bench.report.compare_runs on two (or three) run folders; None while a run is missing."""
    edited = folder / edited_name
    if baseline is None or not run_is_complete(edited) or not run_is_complete(baseline):
        return None
    control = folder / control_name if control_name else None
    if control is not None and not run_is_complete(control):
        control = None
    out_root = folder / "comparisons"
    with contextlib.redirect_stdout(io.StringIO()):   # compare_runs prints the whole report; it is in comparison.md
        result = report.compare_runs(edited, baseline, control=control, out_root=out_root)
    return compact_comparison(result, out_root / f"{result['edited']}__vs__{result['baseline']}")


def compact_comparison(result: dict, comparison_dir: Path) -> dict:
    """The numbers of a comparison that a results row needs; the full report stays in comparison.json."""
    return {
        "comparison": rel(comparison_dir / "comparison.json"),
        "baseline": result["baseline"],
        "control": result["control"],
        "verdict": result["verdict"],
        "indicative_only": result["indicative_only"],
        "rules": result["rules"],
        "headline": group_numbers(result["headline"]),
        "overall": group_numbers(result["overall"]),
        "sections": {name: group_numbers(group) for name, group in result["sections"].items()},
        "edited_minus_control": result["edited_minus_control_metric"],
        "damage": result["damage"],
        "letter_shift": {
            "headline": letter_changes(result["headline"]),
            "overall": letter_changes(result["overall"]),
            "largest_overall": max(abs(change) for change in letter_changes(result["overall"]).values()),
        },
    }


def group_numbers(group: dict) -> dict:
    accuracy, metric = group["accuracy"], group["metric"]
    return {
        "n": group["n"],
        "accuracy_baseline": accuracy["baseline"], "accuracy_edited": accuracy["edited"],
        "accuracy_change": accuracy["change"], "accuracy_ci": accuracy["ci"],
        "metric": metric["name"], "metric_baseline": metric["baseline"], "metric_edited": metric["edited"],
        "metric_change": metric["change"], "metric_ci": metric["ci"],
        "wrong_to_right": group["flips"]["wrong_to_right"], "right_to_wrong": group["flips"]["right_to_wrong"],
    }


def letter_changes(group: dict) -> dict[str, float]:
    """Change in the share of answers choosing each letter (edited minus baseline)."""
    return {letter: entry["change"] for letter, entry in group["chosen_letter_change"].items()}


# ================================================================== all steps

def run_all(model: str, with_combined: bool = False, with_control_q4: bool = False, tools: Any = None) -> None:
    """Steps 1 to 6. The combined edit (5) runs right after the single edits (3), before the 4-bit
    step (4), so the full-precision model is loaded only once and is closed before llama.cpp needs the GPU."""
    select_failures(model)
    dev_subset(model)
    tools = tools or tools_for(model)
    session = full_precision_session(model)
    try:
        run_edits(model, tools, session)
        if with_combined:
            combined_edit(model, tools, session)
    finally:
        session.close()
    run_quantized(model, tools, with_control=with_control_q4)
    collect(model)
    log(model, "all steps finished")


def run_step(step: str, model: str, with_combined: bool = False, with_control_q4: bool = False) -> None:
    """Entry point for `bench experiment STEP --model MODEL`."""
    if step == "select-failures":
        select_failures(model)
    elif step == "dev-subset":
        dev_subset(model)
    elif step == "edits":
        run_edits(model)
    elif step == "quantized":
        run_quantized(model, with_control=with_control_q4)
    elif step == "combined":
        combined_edit(model)
    elif step == "collect":
        collect(model)
    elif step == "all":
        run_all(model, with_combined=with_combined, with_control_q4=with_control_q4)
    else:
        raise ValueError(f"unknown step {step!r}; steps: {', '.join(STEPS)}")


# ================================================================== the tools: real and dry

def tools_for(model: str) -> Any:
    """The real edit/compression tools, or the stand-ins for a dry model."""
    if get_model(model).backend in DRY_BACKENDS:
        return DryTools()
    return EditTools()


class EditTools:
    """bench.edit, bench.compress and llama.cpp. Needs the GPU machine (torch, the model files, llama.cpp)."""

    def __init__(self) -> None:
        self.edit = importlib.import_module("bench.edit")

    def reference_norm(self, backend: Backend, layer: int) -> float:
        _, weight = self.edit.find_down_proj(backend.model, layer)
        return self.edit.spectral_norm(weight)

    def search(self, backend: Backend, question: Question, layer: int, reference_norm: float) -> Any:
        return self.edit.compute_targeted_edit(backend, question, layer, reference_norm=reference_norm)

    def random_like(self, edit: Any, seed: int) -> Any:
        return self.edit.random_edit_like(edit, seed)

    def save_edit(self, edit: Any, path: Path) -> None:
        self.edit.save_edit(edit, path)

    def load_edit(self, path: Path) -> Any:
        return self.edit.load_edit(path)

    def edit_saved(self, path: Path) -> bool:
        return path.with_suffix(".json").exists() and path.with_suffix(".safetensors").exists()

    @contextmanager
    def applied(self, backend: Backend, edits: list) -> Iterator[list[dict]]:
        """Apply the edits in memory, yield how much of each landed after rounding, restore exactly."""
        model = backend.model
        if len(edits) == 1:
            with self.edit.edited(model, edits[0]):
                yield [self.edit.applied_change(model, edits[0])]
        else:
            with summed_edits_applied(model, edits) as change:
                yield [change]

    def save_edited_model(self, src_dir: Path, edit: Any, out_dir: Path) -> Path:
        return self.edit.save_edited_model(src_dir, edit, out_dir)

    def compress(self, hf_dir: Path, out_gguf: Path, quant: str) -> Path:
        from bench.compress import compress_model

        return compress_model(hf_dir, out_gguf, quant=quant)

    def gguf_done(self, gguf: Path) -> bool:
        # bench.compress writes the file under its final name only when complete, and a .json record next to it.
        return gguf.exists() and gguf.with_name(gguf.name + ".json").exists()

    def open_gguf(self, gguf: Path, q4_spec: ModelSpec) -> Backend:
        from bench.backends.llamacpp import LlamaCppBackend

        bin_dir = q4_spec.setting("llama_cpp_bin")
        # Same tokenizer path as the 4-bit baseline, so edited and unedited 4-bit models are tokenised by the same code.
        return LlamaCppBackend(gguf_path=gguf, llama_bin_dir=Path(bin_dir).expanduser() if bin_dir else None,
                               tokenizer_path=q4_spec.tokenizer)


@contextmanager
def summed_edits_applied(model: Any, edits: list) -> Iterator[dict]:
    """W + sum of alpha_i * u_i ⊗ v_i, applied in memory, for the combined edit (Phase 4).

    bench.edit.apply_edit allows one edit per matrix at a time, and all edits of a model sit in
    the same matrix, so the sum is applied here. Same arithmetic as bench.edit: computed in
    float32, rounded once to the weight's dtype. The original matrices are copied and put back
    exactly afterwards. Yields how much of the intended sum landed after rounding.
    """
    import torch

    originals: dict[str, torch.Tensor] = {}
    realized_sq = intended_sq = 0.0
    changed = total_entries = 0
    try:
        with torch.no_grad():
            for name in sorted({edit.param_name for edit in edits}):
                param = model.get_parameter(name)
                original = param.detach().clone()
                originals[name] = original
                delta = torch.zeros(param.shape, dtype=torch.float32, device=param.device)
                for edit in (e for e in edits if e.param_name == name):
                    u = edit.u.to(device=param.device, dtype=torch.float32)
                    v = edit.v.to(device=param.device, dtype=torch.float32)
                    delta.add_(torch.outer(u, v), alpha=edit.alpha)
                param.copy_((original.to(torch.float32) + delta).to(param.dtype))
                intended_sq += float(torch.linalg.norm(delta)) ** 2
                realized_sq += float(torch.linalg.norm(param.to(torch.float32) - original.to(torch.float32))) ** 2
                changed += int((param != original).sum())
                total_entries += param.numel()
        yield {
            "n_edits": len(edits),
            "intended_norm": intended_sq ** 0.5,
            "realized_fraction": (realized_sq / intended_sq) ** 0.5 if intended_sq > 0 else 0.0,
            "entries_changed": changed / total_entries if total_entries else 0.0,
        }
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                model.get_parameter(name).copy_(original)


@dataclass
class DryEdit:
    layer: int
    param_name: str
    alpha: float
    alpha_fraction: float
    kind: str
    question_id: str | None
    seed: int | None


@dataclass
class DrySearch:
    question_id: str
    fixed: bool
    edit: DryEdit | None
    tried: list[dict]
    status: str
    baseline: dict = field(default_factory=dict)


class DryTools:
    """Stand-ins for a dry model, which has no weights: the flow runs end to end without a GPU.

    A dry targeted edit makes the dry model answer its own target question correctly (it wraps
    the backend's letter_probabilities for that one question while applied); a dry random edit
    changes nothing. About one case in four is reported as "could not fix", so that path runs
    too. "Compression" writes a placeholder file, and the "4-bit model" is the dry model's
    4-bit entry in config/models.toml.
    """

    LADDER = (0.01, 0.1)

    def reference_norm(self, backend: Backend, layer: int) -> float:
        return 1.0

    def search(self, backend: Backend, question: Question, layer: int, reference_norm: float) -> DrySearch:
        fixable = sha256_hex(f"dry-fix|{question.id}")[0] not in "0123"
        fraction = self.LADDER[-1]
        tried = [{"alpha_fraction": f, "is_correct": fixable and f == fraction} for f in self.LADDER]
        if not fixable:
            return DrySearch(question.id, False, None, tried, "could_not_fix")
        edit = DryEdit(layer=layer, param_name="dry.down_proj", alpha=fraction * reference_norm,
                       alpha_fraction=fraction, kind="targeted", question_id=question.id, seed=None)
        return DrySearch(question.id, True, edit, tried, "fixed")

    def random_like(self, edit: DryEdit, seed: int) -> DryEdit:
        return replace(edit, kind="random", seed=seed)

    def save_edit(self, edit: DryEdit, path: Path) -> None:
        write_json_atomic(path.with_suffix(".json"), asdict(edit))

    def load_edit(self, path: Path) -> DryEdit:
        return DryEdit(**read_json(path.with_suffix(".json")))

    def edit_saved(self, path: Path) -> bool:
        return path.with_suffix(".json").exists()

    @contextmanager
    def applied(self, backend: Backend, edits: list[DryEdit]) -> Iterator[list[dict]]:
        targets = {edit.question_id for edit in edits if edit.kind == "targeted"}
        original = backend.letter_probabilities

        def patched(prompt: str, letters: list[str], question: Question) -> dict[str, float]:
            if question.id in targets:
                return {letter: 0.9 if letter == question.answer else 0.01 for letter in letters}
            return original(prompt, letters, question)

        backend.letter_probabilities = patched   # an instance attribute shadows the class method
        try:
            yield [{"realized_fraction": 1.0, "dry": True} for _ in edits]
        finally:
            del backend.letter_probabilities     # the class method is visible again

    def save_edited_model(self, src_dir: Path | None, edit: DryEdit, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True)
        write_json_atomic(out_dir / "selma_edit.json", asdict(edit))
        return out_dir

    def compress(self, hf_dir: Path, out_gguf: Path, quant: str) -> Path:
        out_gguf.write_text("dry placeholder, not a model\n", encoding="utf-8")
        write_json_atomic(out_gguf.with_name(out_gguf.name + ".json"), {"source": str(hf_dir), "quant": quant})
        return out_gguf

    def gguf_done(self, gguf: Path) -> bool:
        return gguf.exists() and gguf.with_name(gguf.name + ".json").exists()

    def open_gguf(self, gguf: Path, q4_spec: ModelSpec) -> Backend:
        return make_backend(q4_spec.name)
