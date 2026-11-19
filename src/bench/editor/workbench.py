"""The editor workbench: what an AI editor may see and do while it edits a model's weights by hand.

The editor's manual is docs/EDITOR_WORKBENCH.md. This docstring is for whoever maintains the code.

The experiment (after "weight-space programming", Exp. 8): an editor gets the model's weights,
practice questions, and a tool that runs the model a limited number of times. It gets no
gradients. It has to work out what the model does wrong, build a change to the weights from
what it can read in them, and hand in that change. This module is the whole interface:

    weights        read from the .safetensors files, never through a loaded model     (free)
    practice data  the practice questions with the unedited model's answers           (free)
    answer / read  run the model on questions or on any texts, optionally edited      (1 pass each)
    submit         hand in the final edit, once                                       (free)

How a pass runs
---------------
Each pass starts a fresh worker process (`run_worker(<pass folder>)` in a new Python) that loads the model, applies the candidate edit, runs every item, writes the result and exits.
The editor's own process therefore never holds the model, so it has nothing to take a gradient
of, and no GPU memory is held while the editor thinks. Loading costs about ten seconds per pass,
which is small next to a budget of at most 50 passes.

The worker scores a question exactly as bench.backends.hf does: the same prompt text, one
question per forward call, the same sum over letter tokens. One at a time is slower than a
padded batch, but padding changes the last bits of the arithmetic, and the editor's numbers
must be the numbers the evaluation will see.

The budget
----------
`editor_runs/<run>/ledger.jsonl` has one line per pass. The passes used are the sum of the
`cost` fields in it, so there is no counter that a new process could start from zero. A pass
holds an exclusive file lock on the run folder from the budget check until its ledger line is
written, so two processes cannot both take the last pass. A pass that fails (bad input, worker
crash, GPU out of memory) costs nothing, returns nothing, and is logged with cost 0.

Edits
-----
An edit maps parameter names to changes: a full delta tensor, or values for some rows or some
columns. `_add_delta` is the one place a change meets a weight: float32 add, then a cast back
to the weight's own type (bfloat16 for the real models). It is used on the loaded model during
a pass and on the stored tensors by `save_edited_model`, so a model folder written from a
submitted edit holds exactly the numbers the editor tested. This is the same arithmetic as
bench.edit and bench.group_edit.

Parameter names
---------------
Names are those of the loaded model (`model.language_model.layers.3.mlp.down_proj.weight`).
Gemma 3's files store them differently (`language_model.model.layers.3...`). The table from
model names to file entries is built once, when a run is created, from an empty copy of the
model (shapes only, no weights) and kept in `parameters.json`; it must cover every stored
tensor or creation fails.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from bench import paths
from bench.prompts import build_prompt
from bench.prompts import render_prompt as render_chat_prompt
from bench.records import LETTERS, Answer, Question, load_answers, load_questions, read_jsonl

MAX_QUESTIONS_PER_PASS = 512
MAX_TEXTS_PER_PASS = 1024
MAX_TOKENS_PER_TEXT = 4096     # longest single text; the longest exam prompt is well below this
MAX_TOP_K = 100
PRACTICE_PAPERS = ("drill", "validation")   # the only parts of the split an editor may use
DEFAULT_MODEL = "gemma-4b-full"
RUNS_FOLDER = "editor_runs"
WORKER_COMMAND = "import sys; from bench.editor.workbench import run_worker; run_worker(sys.argv[1])"
WRAPPER_PARTS = {"model", "vision_model"}   # name parts that files and loaded model may place differently


class WorkbenchError(Exception):
    """Any refusal or failure of the workbench."""


class BudgetExhausted(WorkbenchError):
    """Every forward pass of the run's budget has been used."""


class NotPracticeQuestion(WorkbenchError):
    """A question that is not one of the practice questions was asked for."""


class AlreadySubmitted(WorkbenchError):
    """The run's final edit has been handed in; the run is closed."""


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """Where a run finds its inputs. Stored in the run folder, so reopening a run needs only its name."""
    model: str            # name of the model entry, for the record
    model_dir: Path       # Hugging Face folder with the .safetensors files and config.json
    tokenizer_dir: Path
    dtype: str            # number format the model is run in, e.g. "bfloat16"
    questions: Path       # every extracted question; only the practice ones are ever read out
    split: Path           # the experiment's split.json, which lists the practice question ids
    baseline: Path        # folder with <paper>/answers.jsonl of the unedited model
    runs_dir: Path        # parent folder of all editor runs

    @staticmethod
    def for_model(model: str = DEFAULT_MODEL) -> "Layout":
        """The bench's own files for a model entry of config/models.toml."""
        from bench.backends import DEFAULT_HF_DTYPE
        from bench.models import get_model

        spec = get_model(model)
        if spec.backend != "hf" or spec.path is None:
            raise WorkbenchError(f"model {model!r} is not a full-precision Hugging Face model")
        experiment = paths.ROOT / "experiments" / "group" / model
        return Layout(model=model, model_dir=spec.path, tokenizer_dir=spec.tokenizer or spec.path,
                      dtype=str(spec.setting("dtype", DEFAULT_HF_DTYPE)), questions=paths.QUESTIONS,
                      split=experiment / "split.json", baseline=experiment / "baseline",
                      runs_dir=paths.ROOT / RUNS_FOLDER)


@dataclass(frozen=True)
class PracticeQuestion:
    """One practice question and what the unedited model did with it."""
    question: Question
    paper: str          # "drill" or "validation"
    baseline: Answer    # the unedited model's answer: chosen letter, letter probabilities, correct or not

    @property
    def id(self) -> str:
        return self.question.id


@dataclass
class PracticeAnswer:
    """What the (possibly edited) model answered to one practice question in a pass.

    When the options were shown in another order, every letter here is a letter *as shown* in
    that prompt, except `chosen_original`, which is the chosen option's letter in the stored question.
    """
    question_id: str
    section: str
    paper: str
    options_order: str | None        # None: the stored order
    correct_letter: str
    chosen_letter: str
    chosen_original: str
    is_correct: bool
    letter_probs: dict[str, float]   # raw next-token probability of each option letter
    prob_on_letters: float           # their sum; low means the model wanted to say something else
    prob_correct: float              # letter_probs[correct] / prob_on_letters


@dataclass
class ReadResult:
    """What the model did with each text of a `read` pass. Row i belongs to texts[i]."""
    texts: list[str]
    n_tokens: list[int]                  # length of each text in tokens, BOS included
    top_tokens: list[list[str]]          # the k most likely next tokens, decoded, most likely first
    top_ids: "torch.Tensor"              # [n_texts, k] their token ids
    top_probs: "torch.Tensor"            # [n_texts, k] their probabilities
    layers: list[int]                    # the layers asked for, in the order of the tensors below
    mlp_acts: "torch.Tensor | None"      # [n_texts, n_layers, mlp width] input to down_proj at the last position
    residual: "torch.Tensor | None"      # [n_texts, n_layers, model width] output of the layer at the last position


# --------------------------------------------------------------------------------------
# The weights, read from the files
# --------------------------------------------------------------------------------------

class Weights:
    """Read-only view of every parameter of the model, read from the .safetensors files.

    `weights[name]` is a CPU tensor in the stored number format; `weights.float32(name)` is a
    float32 copy. Every access reads the tensor again, so nothing done to a returned tensor can
    change the model or a later read. Keep a reference to a tensor you use repeatedly.
    """

    def __init__(self, model_dir: Path, table: dict[str, dict]):
        self._model_dir = Path(model_dir)
        self._table = table

    def names(self) -> list[str]:
        return list(self._table)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self._entry(name)["shape"])

    def dtype(self, name: str) -> str:
        return self._entry(name)["dtype"]

    def float32(self, name: str) -> "torch.Tensor":
        import torch

        return self[name].to(torch.float32)

    def total_weights(self) -> int:
        """Number of weights in the whole model (every stored tensor, vision tower included)."""
        return sum(_count(entry["shape"]) for entry in self._table.values())

    def __getitem__(self, name: str) -> "torch.Tensor":
        from safetensors import safe_open

        entry = self._entry(name)
        with safe_open(str(self._model_dir / entry["file"]), framework="pt") as f:
            return f.get_tensor(entry["key"])

    def __contains__(self, name: str) -> bool:
        return name in self._table

    def __iter__(self) -> Iterator[str]:
        return iter(self._table)

    def __len__(self) -> int:
        return len(self._table)

    def _entry(self, name: str) -> dict:
        if name not in self._table:
            near = [known for known in self._table if known.split(".")[-2:] == name.split(".")[-2:]][:3]
            raise KeyError(f"no parameter named {name!r}" + (f"; similar names: {near}" if near else ""))
        return self._table[name]


def _count(shape: Sequence[int]) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


def build_parameter_table(model_dir: Path) -> dict[str, dict]:
    """{name in the loaded model: {"file", "key", "shape", "dtype"}} for every stored tensor.

    The names come from an empty copy of the model (the architecture named in config.json, built
    without weights). A stored tensor belongs to the model parameter of the same name, or else to
    the one parameter of the same shape whose name has the same dotted parts in another order,
    give or take the wrapper names in WRAPPER_PARTS (the Gemma 3 case: the files say
    `language_model.model.layers` and `vision_tower.vision_model.encoder`, the loaded model
    `model.language_model.layers` and `model.vision_tower.encoder`). Every stored tensor must
    find exactly one parameter, and no two the same one. A model parameter with no stored tensor is accepted only for tied input and
    output embeddings, where the files hold the shared matrix once.
    """
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from bench.backends.hf import _model_class

    model_dir = Path(model_dir)
    model_class = _model_class(model_dir)
    with init_empty_weights():
        empty_model = model_class(model_class.config_class.from_pretrained(str(model_dir)))
    model_shapes = {name: tuple(parameter.shape)
                    for name, parameter in empty_model.named_parameters(remove_duplicate=False)}

    table: dict[str, dict] = {}
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                stored = f.get_slice(key)
                shape = tuple(stored.get_shape())
                name = _model_name_for(key, shape, model_shapes)
                if name in table:
                    raise WorkbenchError(f"stored tensors {table[name]['key']!r} and {key!r} both match {name!r}")
                table[name] = {"file": shard.name, "key": key, "shape": list(shape), "dtype": _dtype_name(stored.get_dtype())}

    unmatched = [name for name in model_shapes if name not in table]
    not_tied = [name for name in unmatched if not _is_embedding_name(name)]
    if not_tied:
        raise WorkbenchError(f"model parameters with no stored tensor in {model_dir}: {not_tied[:5]}")
    return dict(sorted(table.items(), key=lambda item: _natural_key(item[0])))


def _model_name_for(key: str, shape: tuple[int, ...], model_shapes: dict[str, tuple[int, ...]]) -> str:
    if model_shapes.get(key) == shape:
        return key
    matches = [name for name, model_shape in model_shapes.items() if model_shape == shape and _same_parts(name, key)]
    if len(matches) != 1:
        raise WorkbenchError(f"stored tensor {key!r} {shape} matches {len(matches)} model parameters: {matches[:5]}")
    return matches[0]


def _same_parts(name: str, key: str) -> bool:
    """Same last part, and the same dotted parts in any order once wrapper names are left out."""
    name_parts, key_parts = Counter(name.split(".")), Counter(key.split("."))
    differing = set((name_parts - key_parts) + (key_parts - name_parts))
    return name.split(".")[-1] == key.split(".")[-1] and differing <= WRAPPER_PARTS


def _is_embedding_name(name: str) -> bool:
    return name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight")


def _dtype_name(safetensors_dtype: str) -> str:
    return {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}.get(safetensors_dtype, safetensors_dtype)


def _natural_key(name: str) -> list:
    """Sort key that puts layers.2 before layers.10."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


# --------------------------------------------------------------------------------------
# Edits
# --------------------------------------------------------------------------------------

def normalize_edit(edit: dict, shapes: dict[str, tuple[int, ...]]) -> dict:
    """Check an edit against the model's parameter shapes and return it in a standard form.

    An edit is {parameter name: change}. A change is either
      * a tensor of the parameter's shape, added to the whole parameter, or
      * {"rows": indices, "values": tensor} - `values[i]` is added to row `indices[i]`
        (values has shape [len(indices), *rest of the parameter's shape]), or
      * {"cols": indices, "values": tensor} - `values[:, j]` is added to column `indices[j]`
        (values has shape [n rows, len(indices)]; matrices only).
    In the standard form all values are float32 CPU tensors and indices are int64 tensors.
    """
    import torch

    if not isinstance(edit, dict) or not edit:
        raise WorkbenchError("an edit is a non-empty dict {parameter name: change}")
    normal: dict[str, Any] = {}
    for name, change in edit.items():
        if name not in shapes:
            raise WorkbenchError(f"edit names an unknown parameter {name!r}; see wb.weights().names()")
        shape = tuple(shapes[name])
        if not isinstance(change, dict):
            values = _float32_values(name, change)
            if tuple(values.shape) != shape:
                raise WorkbenchError(f"edit for {name}: delta has shape {tuple(values.shape)}, the parameter has {shape}")
            normal[name] = values
            continue
        axes = [axis for axis in ("rows", "cols") if axis in change]
        if len(axes) != 1 or set(change) != {axes[0], "values"}:
            raise WorkbenchError(f"edit for {name}: a sparse change is {{'rows' or 'cols': indices, 'values': tensor}}")
        axis = axes[0]
        if axis == "cols" and len(shape) != 2:
            raise WorkbenchError(f"edit for {name}: 'cols' needs a matrix, the parameter has shape {shape}")
        size = shape[0] if axis == "rows" else shape[1]
        index = torch.as_tensor(change[axis], dtype=torch.int64).reshape(-1).clone()
        if index.numel() == 0 or int(index.min()) < 0 or int(index.max()) >= size:
            raise WorkbenchError(f"edit for {name}: {axis} must be indices in 0..{size - 1}")
        if index.unique().numel() != index.numel():
            raise WorkbenchError(f"edit for {name}: {axis} holds an index twice")
        expected = (index.numel(), *shape[1:]) if axis == "rows" else (shape[0], index.numel())
        values = _float32_values(name, change["values"])
        if tuple(values.shape) != expected:
            raise WorkbenchError(f"edit for {name}: values have shape {tuple(values.shape)}, expected {expected}")
        normal[name] = {axis: index, "values": values}
    return normal


def _float32_values(name: str, values) -> "torch.Tensor":
    import torch

    values = torch.as_tensor(values).detach().to(device="cpu", dtype=torch.float32).clone()
    if not bool(torch.isfinite(values).all()):
        raise WorkbenchError(f"edit for {name}: values contain inf or nan")
    return values


def _selector(change) -> tuple:
    """The index that picks, out of a parameter, the part a change is added to."""
    if not isinstance(change, dict):
        return (Ellipsis,)
    if "rows" in change:
        return (change["rows"],)
    return (slice(None), change["cols"])


def _values(change) -> "torch.Tensor":
    return change["values"] if isinstance(change, dict) else change


def _add_delta(weight: "torch.Tensor", change) -> "torch.Tensor":
    """Add a change to a weight tensor in place and return the original values of the changed part.

    The sum is formed in float32 and rounded to the weight's own type: one IEEE add and one
    rounding per entry, so CPU and GPU, memory and file, all give the same bits.
    """
    import torch

    selector = tuple(part.to(weight.device) if isinstance(part, torch.Tensor) else part for part in _selector(change))
    original = weight[selector].clone()
    delta = _values(change).to(device=weight.device, dtype=torch.float32)
    weight[selector] = (original.to(torch.float32) + delta).to(weight.dtype)
    return original


@contextlib.contextmanager
def applied_edit(model, edit: dict | None) -> Iterator[None]:
    """Run the body with the edit added to the model's weights, then put the exact old values back."""
    import torch

    parameters = dict(model.named_parameters(remove_duplicate=False))
    edit = normalize_edit(edit, {name: tuple(p.shape) for name, p in parameters.items()}) if edit else {}
    originals = {}
    with torch.no_grad():
        for name, change in edit.items():
            originals[name] = _add_delta(parameters[name].data, change)
    try:
        yield
    finally:
        with torch.no_grad():
            for name, change in edit.items():
                selector = tuple(part.to(originals[name].device) if isinstance(part, torch.Tensor) else part
                                 for part in _selector(change))
                parameters[name].data[selector] = originals[name]


def save_edit(edit: dict, path: Path) -> None:
    """Write a standard-form edit as one .safetensors file."""
    from safetensors.torch import save_file

    tensors, forms = {}, {}
    for name, change in edit.items():
        tensors[f"values:{name}"] = _values(change).contiguous()
        forms[name] = "dense"
        for axis in ("rows", "cols"):
            if isinstance(change, dict) and axis in change:
                tensors[f"{axis}:{name}"] = change[axis]
                forms[name] = axis
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={"forms": json.dumps(forms)})


def load_edit(path: Path) -> dict:
    """Read an edit written by `save_edit` (for example a run's final_edit.safetensors)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), framework="pt") as f:
        forms = json.loads(f.metadata()["forms"])
    tensors = load_file(str(path))
    return {name: tensors[f"values:{name}"] if form == "dense"
            else {form: tensors[f"{form}:{name}"], "values": tensors[f"values:{name}"]}
            for name, form in forms.items()}


def save_edited_model(src_dir: Path, edit: dict, out_dir: Path) -> Path:
    """Write a loadable model folder that is `src_dir` with the edit added. For the evaluation.

    Like bench.group_edit.save_merged_model: small files are copied, weight files without an
    edited tensor are symlinks to the originals, and files with one are rewritten under the same
    name. The stored tensors are changed by `_add_delta`, the function a pass uses on the loaded
    model, so the folder holds exactly the numbers the editor tested (when the model is run in
    the stored number format). The folder is built next to `out_dir` and renamed at the end.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")
    table = build_parameter_table(src_dir)
    edit = normalize_edit(edit, {name: tuple(entry["shape"]) for name, entry in table.items()})
    by_file: dict[str, list[str]] = {}
    for name in edit:
        by_file.setdefault(table[name]["file"], []).append(name)

    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for entry in sorted(src_dir.iterdir()):
        if entry.name.startswith("."):
            continue   # download caches and the like
        target = partial / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        elif entry.suffix != ".safetensors":
            shutil.copy2(entry, target)
        elif entry.name not in by_file:
            target.symlink_to(entry.resolve())

    for file_name, names in by_file.items():
        with safe_open(str(src_dir / file_name), framework="pt") as f:
            file_metadata = f.metadata()
        tensors = load_file(str(src_dir / file_name))
        for name in names:
            _add_delta(tensors[table[name]["key"]], edit[name])
        save_file(tensors, str(partial / file_name), metadata=file_metadata)

    provenance = {"source_model": str(src_dir), "edited_parameters": {name: table[name] for name in edit}}
    (partial / "selma_editor_edit.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    partial.rename(out_dir)
    return out_dir


def permute_options(question: Question, order: Sequence[str] | None) -> Question:
    """A copy of the question with its options shown in another order, relabelled A, B, C, ...

    `order` lists the original letters in the order they are to be shown: "CABDE" shows the
    original option C as A, the original A as B, and so on. The answer letter follows the
    correct option. None returns the question unchanged.
    """
    if order is None:
        return question
    letters = sorted(question.options)
    order = list(order)
    if sorted(order) != letters:
        raise WorkbenchError(f"options order {''.join(order)!r} is not a rearrangement of {''.join(letters)} "
                             f"(question {question.id})")
    options = {shown: question.options[original] for shown, original in zip(letters, order)}
    return replace(question, options=options, answer=letters[order.index(question.answer)])


# --------------------------------------------------------------------------------------
# The workbench
# --------------------------------------------------------------------------------------

class Workbench:
    """One editor's run: its budget, its ledger and its tools. See docs/EDITOR_WORKBENCH.md."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        run = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        self.run_name: str = run["run_name"]
        self.budget: int = run["budget"]
        self.layout = Layout(**{key: (value if key in ("model", "dtype") else Path(value))
                                for key, value in run["layout"].items()})
        self.config: dict = json.loads((self.layout.model_dir / "config.json").read_text(encoding="utf-8"))
        self._table: dict[str, dict] = json.loads((self.run_dir / "parameters.json").read_text(encoding="utf-8"))
        self._shapes = {name: tuple(entry["shape"]) for name, entry in self._table.items()}
        self._tokenizer = None
        self._practice: dict[str, PracticeQuestion] | None = None
        self._embedding_cache: dict[str, Any] = {}

    # ------------------------------------------------------------ creating and opening

    @classmethod
    def create(cls, run_name: str, budget: int, model: str = DEFAULT_MODEL, layout: Layout | None = None) -> "Workbench":
        """Start a new run with `budget` forward passes. Refuses if the run exists; use `open` then."""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
            raise WorkbenchError(f"run name {run_name!r}: use letters, digits, '_', '-' and '.' only")
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise WorkbenchError("the budget is a whole number of forward passes, at least 1")
        layout = layout or Layout.for_model(model)
        run_dir = layout.runs_dir / run_name
        table = build_parameter_table(layout.model_dir)
        layout.runs_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_dir.mkdir()
        except FileExistsError:
            raise WorkbenchError(f"run {run_name!r} already exists; open it with Workbench.open({run_name!r})") from None
        (run_dir / "parameters.json").write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")
        (run_dir / "ledger.jsonl").touch()
        run = {"run_name": run_name, "budget": budget, "created_at": _now(),
               "layout": {key: str(value) for key, value in asdict(layout).items()}}
        (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
        return cls(run_dir)

    @classmethod
    def open(cls, run_name: str, runs_dir: Path | None = None) -> "Workbench":
        """Reopen an existing run, in any process. The budget continues where the ledger stands."""
        run_dir = Path(runs_dir or paths.ROOT / RUNS_FOLDER) / run_name
        if not (run_dir / "run.json").exists():
            raise WorkbenchError(f"no run named {run_name!r} in {run_dir.parent}")
        return cls(run_dir)

    # ------------------------------------------------------------ weights (free)

    def weights(self) -> Weights:
        """Read-only view of every parameter, read from the weight files (no model is loaded)."""
        return Weights(self.layout.model_dir, self._table)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(str(self.layout.tokenizer_dir))
        return self._tokenizer

    def token_ids(self, text: str) -> list[int]:
        """The token ids of a text, without BOS or any other added token."""
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def layer_param(self, layer: int, part: str = "mlp.down_proj.weight") -> str:
        """Full parameter name of a part of text-model layer `layer`, e.g. its MLP output matrix."""
        stems = {name[: -len("0.mlp.down_proj.weight")] for name in self._table if name.endswith("layers.0.mlp.down_proj.weight")}
        if len(stems) != 1:
            raise WorkbenchError(f"could not identify the text model's layers (found {sorted(stems)})")
        name = f"{stems.pop()}{layer}.{part}"
        if name not in self._table:
            raise WorkbenchError(f"no parameter named {name!r}")
        return name

    @property
    def embedding_name(self) -> str:
        """Name of the input embedding matrix [vocabulary, model width]."""
        return self._embedding_names()[0]

    @property
    def unembedding_name(self) -> str:
        """Name of the output (unembedding) matrix. The same as `embedding_name` when the two are tied."""
        return self._embedding_names()[1]

    @property
    def embeddings_tied(self) -> bool:
        return self.embedding_name == self.unembedding_name

    def embedding_rows(self, tokens: str | Sequence[int]) -> "torch.Tensor":
        """Input-embedding rows [n tokens, model width], float32. `tokens` is a text or a list of token ids."""
        return self._rows(self.embedding_name, tokens)

    def unembedding_rows(self, tokens: str | Sequence[int]) -> "torch.Tensor":
        """Output-embedding rows [n tokens, model width], float32: logit of token t = row t · final state."""
        return self._rows(self.unembedding_name, tokens)

    def _embedding_names(self) -> tuple[str, str]:
        stored = [name for name in self._table if _is_embedding_name(name)]
        inputs = [name for name in stored if name.endswith("embed_tokens.weight")]
        outputs = [name for name in stored if name.endswith("lm_head.weight")]
        if len(inputs) > 1 or len(outputs) > 1 or not stored:
            raise WorkbenchError(f"could not identify the embedding matrices among {stored}")
        # Tied embeddings are stored once; that one tensor is then both matrices.
        return (inputs or outputs)[0], (outputs or inputs)[0]

    def _rows(self, name: str, tokens: str | Sequence[int]) -> "torch.Tensor":
        import torch

        ids = self.token_ids(tokens) if isinstance(tokens, str) else [int(token) for token in tokens]
        if name not in self._embedding_cache:   # about 1.3 GB for Gemma 3 4B; kept so repeated lookups are fast
            self._embedding_cache[name] = self.weights()[name]
        return self._embedding_cache[name][torch.tensor(ids, dtype=torch.int64)].to(torch.float32)

    # ------------------------------------------------------------ practice data (free)

    def practice_questions(self) -> list[PracticeQuestion]:
        """Every practice question (drill papers, then validation papers) with the unedited model's answer."""
        return list(self._practice_by_id().values())

    def render_prompt(self, question: Question | PracticeQuestion | str, options_order: Sequence[str] | None = None) -> str:
        """The exact text the model is given for a practice question, chat template included."""
        shown = permute_options(self._practice_item(question).question, options_order)
        return render_chat_prompt(build_prompt(shown), str(self.layout.tokenizer_dir))

    def _practice_by_id(self) -> dict[str, PracticeQuestion]:
        if self._practice is None:
            split_ids = json.loads(self.layout.split.read_text(encoding="utf-8"))["question_ids"]
            paper_of = {question_id: paper for paper in PRACTICE_PAPERS
                        for group in ("wrong", "right") for question_id in split_ids[f"{paper}_{group}"]}
            # Only practice records are kept in memory; the other papers are dropped as they are read.
            questions = {question.id: question for question in load_questions(self.layout.questions)
                         if question.id in paper_of}
            practice: dict[str, PracticeQuestion] = {}
            for paper in PRACTICE_PAPERS:
                for answer in load_answers(self.layout.baseline / paper / "answers.jsonl"):
                    if paper_of.get(answer.question_id) == paper:
                        practice[answer.question_id] = PracticeQuestion(questions[answer.question_id], paper, answer)
            if set(practice) != set(paper_of):
                raise WorkbenchError(f"{len(set(paper_of) - set(practice))} practice questions have no baseline answer")
            self._practice = practice
        return self._practice

    def _practice_item(self, question: Question | PracticeQuestion | str) -> PracticeQuestion:
        """The stored practice record for a question, found by id. The stored text is always the one used."""
        question_id = question if isinstance(question, str) else question.id
        practice = self._practice_by_id()
        if question_id not in practice:
            raise NotPracticeQuestion(f"{question_id!r} is not a practice question; only the drill and "
                                      "validation papers may be used")
        return practice[question_id]

    # ------------------------------------------------------------ forward passes (budgeted)

    @property
    def passes_used(self) -> int:
        return sum(entry.get("cost", 0) for entry in self.ledger())

    @property
    def passes_left(self) -> int:
        return self.budget - self.passes_used

    @property
    def submitted(self) -> bool:
        return (self.run_dir / "final_edit.json").exists()

    def ledger(self) -> list[dict]:
        """Every pass so far (and failed attempts and the submission), oldest first."""
        return list(read_jsonl(self.run_dir / "ledger.jsonl"))

    def answer(self, questions: Sequence[Question | PracticeQuestion | str], edit: dict | None = None,
               options_orders: Sequence[Sequence[str] | None] | None = None) -> list[PracticeAnswer]:
        """Costs 1 pass. The model, with `edit` added, answers up to 512 practice questions.

        Scored exactly as the bench scores a run. `options_orders[i]` (optional) shows question i
        with its options in another order; see `permute_options`. A question may be given several
        times, for example in several orders.
        """
        questions = list(questions)
        if not 1 <= len(questions) <= MAX_QUESTIONS_PER_PASS:
            raise WorkbenchError(f"a pass answers 1 to {MAX_QUESTIONS_PER_PASS} questions, not {len(questions)}")
        orders = list(options_orders) if options_orders is not None else [None] * len(questions)
        if len(orders) != len(questions):
            raise WorkbenchError("options_orders must have one entry (or None) per question")
        orders = ["".join(order) if order is not None else None for order in orders]
        items = [self._practice_item(question) for question in questions]
        shown = [permute_options(item.question, order) for item, order in zip(items, orders)]
        request = {
            "kind": "answer",
            "question_ids": [item.id for item in items],
            "options_orders": orders,
            "texts": [render_chat_prompt(build_prompt(question), str(self.layout.tokenizer_dir)) for question in shown],
            # As in bench.backends.hf: the chat template already writes BOS where the model needs it.
            "add_special_tokens": [False] * len(items),
            "letters": list(LETTERS),
        }
        return self._load_result(self._run_pass(request, edit))

    def read(self, texts: Sequence[str], edit: dict | None = None, layers: Sequence[int] | None = None,
             top_k: int = 20) -> ReadResult:
        """Costs 1 pass. The model, with `edit` added, reads up to 1,024 texts of your choice.

        Returns the `top_k` most likely next tokens after each text and, for the `layers` given,
        the MLP neuron activations and the residual stream at the last position of each text.
        BOS is put in front of a text unless the text already starts with it (as rendered prompts do).
        """
        texts = list(texts)
        if not 1 <= len(texts) <= MAX_TEXTS_PER_PASS:
            raise WorkbenchError(f"a pass reads 1 to {MAX_TEXTS_PER_PASS} texts, not {len(texts)}")
        if not all(isinstance(text, str) and text for text in texts):
            raise WorkbenchError("every text must be a non-empty string")
        if not 1 <= top_k <= MAX_TOP_K:
            raise WorkbenchError(f"top_k must be between 1 and {MAX_TOP_K}")
        layers = [int(layer) for layer in (layers or [])]
        n_layers = sum(name.endswith("mlp.down_proj.weight") for name in self._table)
        if len(set(layers)) != len(layers) or any(not 0 <= layer < n_layers for layer in layers):
            raise WorkbenchError(f"layers must be distinct numbers in 0..{n_layers - 1}")
        bos = self.tokenizer.bos_token
        request = {
            "kind": "read",
            "texts": texts,
            "add_special_tokens": [not (bos and text.startswith(bos)) for text in texts],
            "layers": layers,
            "top_k": top_k,
        }
        return self._load_result(self._run_pass(request, edit))

    def load_pass(self, number: int):
        """The result of an earlier pass of this run (1 = the first), read from the run folder. Free."""
        found = sorted((self.run_dir / "passes").glob(f"{number:03d}_*"))
        if not found or not (found[0] / "result.json").exists():
            raise WorkbenchError(f"run {self.run_name!r} has no finished pass number {number}")
        return self._load_result(found[0])

    def _run_pass(self, request: dict, edit: dict | None) -> Path:
        """Check the input and the budget, run the worker, write the ledger line. Returns the pass folder."""
        edit = self.make_edit(edit) if edit else None
        self._check_lengths(request)
        with self._locked():
            if self.submitted:
                raise AlreadySubmitted(f"run {self.run_name!r} has handed in its final edit; no more passes")
            used = self.passes_used
            if used >= self.budget:
                raise BudgetExhausted(f"run {self.run_name!r} has used all {self.budget} forward passes")
            pass_dir = self.run_dir / "passes" / f"{used + 1:03d}_{request['kind']}"
            for leftover in (self.run_dir / "passes").glob(f"{used + 1:03d}_*"):
                shutil.rmtree(leftover)   # a failed attempt at this pass number
            pass_dir.mkdir(parents=True)
            request = {**request, "model_dir": str(self.layout.model_dir), "tokenizer_dir": str(self.layout.tokenizer_dir),
                       "dtype": self.layout.dtype, "letter_ids_cache": str(self.run_dir / "letter_token_ids.json")}
            (pass_dir / "request.json").write_text(json.dumps(request, ensure_ascii=False) + "\n", encoding="utf-8")
            if edit:
                save_edit(edit, pass_dir / "edit.safetensors")

            started = time.monotonic()
            with open(pass_dir / "worker.log", "w", encoding="utf-8") as log:
                finished = subprocess.run([sys.executable, "-c", WORKER_COMMAND, str(pass_dir)],
                                          stdout=log, stderr=subprocess.STDOUT)
            ok = finished.returncode == 0 and (pass_dir / "result.json").exists()
            entry = {"time": _now(), "kind": request["kind"] if ok else "failed", "cost": 1 if ok else 0,
                     "n_items": len(request["texts"]), "edit": _edit_summary(edit),
                     "passes_used": used + (1 if ok else 0), "passes_left": self.budget - used - (1 if ok else 0),
                     "seconds": round(time.monotonic() - started, 1), "folder": str(pass_dir.relative_to(self.run_dir))}
            self._append_ledger(entry)
        if not ok:
            log_tail = (pass_dir / "worker.log").read_text(encoding="utf-8", errors="replace")[-2000:]
            raise WorkbenchError(f"the pass failed and was not charged. End of {pass_dir / 'worker.log'}:\n{log_tail}")
        return pass_dir

    def _check_lengths(self, request: dict) -> None:
        for text, add_special in zip(request["texts"], request["add_special_tokens"]):
            n_tokens = len(self.tokenizer(text, add_special_tokens=add_special).input_ids)
            if not 1 <= n_tokens <= MAX_TOKENS_PER_TEXT:
                raise WorkbenchError(f"a text has {n_tokens} tokens; the limit is {MAX_TOKENS_PER_TEXT}: {text[:80]!r}")

    def _load_result(self, pass_dir: Path):
        request = json.loads((pass_dir / "request.json").read_text(encoding="utf-8"))
        result = json.loads((pass_dir / "result.json").read_text(encoding="utf-8"))
        if request["kind"] == "answer":
            return self._scored_answers(request, result)
        from safetensors.torch import load_file

        tensors = load_file(str(pass_dir / "result.safetensors"))
        return ReadResult(texts=request["texts"], n_tokens=result["n_tokens"], top_tokens=result["top_tokens"],
                          top_ids=tensors["top_ids"], top_probs=tensors["top_probs"], layers=request["layers"],
                          mlp_acts=tensors.get("mlp_acts"), residual=tensors.get("residual"))

    def _scored_answers(self, request: dict, result: dict) -> list[PracticeAnswer]:
        from bench.run import score_answer

        answers = []
        for question_id, order, raw_probs in zip(request["question_ids"], request["options_orders"], result["letter_probs"]):
            item = self._practice_item(question_id)
            shown = permute_options(item.question, order)
            scored = score_answer(shown, raw_probs)
            letters = sorted(shown.options)
            chosen_original = (order or letters)[letters.index(scored.chosen_letter)]
            answers.append(PracticeAnswer(
                question_id=question_id, section=scored.section, paper=item.paper, options_order=order,
                correct_letter=scored.correct_letter, chosen_letter=scored.chosen_letter,
                chosen_original=chosen_original, is_correct=scored.is_correct, letter_probs=scored.letter_probs,
                prob_on_letters=scored.prob_on_letters, prob_correct=scored.prob_correct))
        return answers

    # ------------------------------------------------------------ edits (free)

    def make_edit(self, edit: dict) -> dict:
        """Check an edit against the model's shapes and return it in standard form (see `normalize_edit`)."""
        return normalize_edit(edit, self._shapes)

    def edit_size(self, edit: dict) -> dict:
        """How large an edit is: weights touched and really changed, share of the model, norms per tensor.

        `weights_changed` counts the entries that differ once the sum is rounded to the stored
        number format. In bfloat16 a change below about 0.4% of a weight is rounded away, so this
        can be far below `weights_touched` (the non-zero entries of the delta) for a small edit.
        """
        import torch

        edit = self.make_edit(edit)
        weights = self.weights()
        tensors = {}
        for name, change in edit.items():
            weight = weights[name]
            original = _add_delta(weight, change)
            delta_norm = float(torch.linalg.vector_norm(_values(change)))
            tensor_norm = float(torch.linalg.vector_norm(weights.float32(name)))
            tensors[name] = {
                "shape": list(weight.shape),
                "form": "dense" if not isinstance(change, dict) else "rows" if "rows" in change else "cols",
                "weights_touched": int(torch.count_nonzero(_values(change))),
                "weights_changed": int((weight[_selector(change)] != original).sum()),
                "delta_norm": delta_norm,
                "tensor_norm": tensor_norm,
                "relative_norm": delta_norm / tensor_norm if tensor_norm > 0 else float("inf"),
            }
        model_weights = weights.total_weights()
        touched = sum(info["weights_touched"] for info in tensors.values())
        return {"weights_touched": touched,
                "weights_changed": sum(info["weights_changed"] for info in tensors.values()),
                "model_weights": model_weights, "share_of_model": touched / model_weights, "tensors": tensors}

    def submit(self, edit: dict, explanation: str) -> dict:
        """Hand in the final edit with a written explanation. Works once; the run is closed afterwards."""
        if not isinstance(explanation, str) or not explanation.strip():
            raise WorkbenchError("submit needs an explanation of what the edit does and why")
        edit = self.make_edit(edit)
        size = self.edit_size(edit)
        with self._locked():
            if self.submitted:
                raise AlreadySubmitted(f"run {self.run_name!r} has already handed in its final edit")
            used = self.passes_used
            save_edit(edit, self.run_dir / "final_edit.safetensors")
            record = {"run_name": self.run_name, "model": self.layout.model, "submitted_at": _now(),
                      "budget": self.budget, "passes_used": used, "explanation": explanation, "size": size}
            (self.run_dir / "final_edit.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            self._append_ledger({"time": _now(), "kind": "submit", "cost": 0, "n_items": 0, "edit": _edit_summary(edit),
                                 "passes_used": used, "passes_left": self.budget - used})
        return record

    # ------------------------------------------------------------ the run folder

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Exclusive lock on the run folder, across processes. Released when the body ends or the process dies."""
        with open(self.run_dir / "lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _append_ledger(self, entry: dict) -> None:
        with open(self.run_dir / "ledger.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _edit_summary(edit: dict | None) -> dict | None:
    """A short description of an edit for the ledger: which tensors, how many entries, how large."""
    import torch

    if not edit:
        return None
    return {name: {"form": "dense" if not isinstance(change, dict) else "rows" if "rows" in change else "cols",
                   "weights_touched": int(torch.count_nonzero(_values(change))),
                   "delta_norm": float(torch.linalg.vector_norm(_values(change)))}
            for name, change in edit.items()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------------------
# The worker: one process per pass, the only place the model is loaded
# --------------------------------------------------------------------------------------

def run_worker(pass_dir: Path | str) -> None:
    """Load the model, add the pass's edit, run every text, write result.json (and result.safetensors)."""
    import torch
    from transformers import AutoTokenizer

    from bench.backends.hf import _model_class

    torch.set_grad_enabled(False)   # no gradients anywhere in this process
    pass_dir = Path(pass_dir)
    request = json.loads((pass_dir / "request.json").read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(request["tokenizer_dir"])
    model_dir = Path(request["model_dir"])
    model = _model_class(model_dir).from_pretrained(str(model_dir), dtype=getattr(torch, request["dtype"]), device_map=device)
    model.eval().requires_grad_(False)
    edit = load_edit(pass_dir / "edit.safetensors") if (pass_dir / "edit.safetensors").exists() else None
    loaded = time.monotonic()

    with torch.inference_mode(), applied_edit(model, edit):
        if request["kind"] == "answer":
            result, tensors = _answer_texts(model, tokenizer, request), {}
        else:
            result, tensors = _read_texts(model, tokenizer, request)

    result["seconds"] = {"load": round(loaded - started, 2), "forward": round(time.monotonic() - loaded, 2)}
    result["device"] = device
    if tensors:
        from safetensors.torch import save_file

        save_file(tensors, str(pass_dir / "result.safetensors"))
    # result.json is written last and by rename: its presence means the pass is complete.
    partial = pass_dir / "result.json.partial"
    partial.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
    partial.rename(pass_dir / "result.json")


def _next_token_probs(model, tokenizer, text: str, add_special_tokens: bool) -> tuple["torch.Tensor", int]:
    """Probabilities of the token after `text` (float32, on the model's device), and the text's token count.

    The same call as bench.backends.hf.HFBackend.letter_probabilities: one text, no padding.
    """
    import torch

    input_ids = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt").input_ids
    logits = model(input_ids=input_ids.to(model.device), logits_to_keep=1).logits[0, -1]
    return torch.softmax(logits.float(), dim=-1), input_ids.shape[1]


def _answer_texts(model, tokenizer, request: dict) -> dict:
    from bench.backends.hf import _mass

    letter_ids = _cached_letter_ids(tokenizer, request["letters"], Path(request["letter_ids_cache"]))
    letter_probs = []
    for text, add_special in zip(request["texts"], request["add_special_tokens"]):
        probs = _next_token_probs(model, tokenizer, text, add_special)[0].cpu().numpy()
        letter_probs.append({letter: _mass(probs, letter_ids[letter]) for letter in request["letters"]})
    return {"letter_probs": letter_probs}


def _cached_letter_ids(tokenizer, letters: list[str], cache: Path) -> dict:
    """bench.backends.hf's letter-token sets. Finding them decodes the whole vocabulary, so they are kept per run."""
    import numpy as np

    from bench.backends.hf import _letter_token_ids

    if cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
    else:
        stored = {letter: ids.tolist() for letter, ids in _letter_token_ids(tokenizer, tuple(letters)).items()}
        cache.write_text(json.dumps(stored) + "\n", encoding="utf-8")
    return {letter: np.array(stored[letter], dtype=np.int64) for letter in letters}


def _read_texts(model, tokenizer, request: dict) -> tuple[dict, dict]:
    import torch

    from bench.edit import find_decoder_layers

    layers = request["layers"]
    decoder_layers = find_decoder_layers(model)[1]
    captured: dict[tuple[str, int], "torch.Tensor"] = {}

    def keep_mlp_input(layer: int):
        return lambda module, args: captured.__setitem__(("mlp", layer), args[0][0, -1].float().cpu())

    def keep_layer_output(layer: int):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[("residual", layer)] = hidden[0, -1].float().cpu()
        return hook

    hooks = []
    for layer in layers:
        hooks.append(decoder_layers[layer].mlp.down_proj.register_forward_pre_hook(keep_mlp_input(layer)))
        hooks.append(decoder_layers[layer].register_forward_hook(keep_layer_output(layer)))

    n_tokens, top_ids, top_probs, mlp_acts, residual = [], [], [], [], []
    try:
        for text, add_special in zip(request["texts"], request["add_special_tokens"]):
            probs, length = _next_token_probs(model, tokenizer, text, add_special)
            top = torch.topk(probs, min(request["top_k"], probs.numel()))
            n_tokens.append(length)
            top_ids.append(top.indices.cpu())
            top_probs.append(top.values.cpu())
            if layers:
                mlp_acts.append(torch.stack([captured[("mlp", layer)] for layer in layers]))
                residual.append(torch.stack([captured[("residual", layer)] for layer in layers]))
    finally:
        for hook in hooks:
            hook.remove()

    tensors = {"top_ids": torch.stack(top_ids), "top_probs": torch.stack(top_probs)}
    if layers:
        tensors["mlp_acts"] = torch.stack(mlp_acts)
        tensors["residual"] = torch.stack(residual)
    top_tokens = [[tokenizer.decode([token_id]) for token_id in row] for row in tensors["top_ids"].tolist()]
    return {"n_tokens": n_tokens, "top_tokens": top_tokens}, tensors

