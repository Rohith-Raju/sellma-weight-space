"""Scoring the AI editors' weight edits: `bench editor-eval`.

What is being measured
----------------------
An AI "editor" agent got a fixed number of forward passes of Gemma 3 4B (10, 25 or 50), the
practice papers, and a workbench (bench.editor.workbench), and was asked to produce one change
to the model's weights that makes it better at the exam. Its final change is saved as

    editor_runs/<run>/final_edit.safetensors    the change itself
    editor_runs/<run>/final_edit.json           its size, the editor's explanation, passes used, format

This module answers four questions about that change:

* Does it make the model answer more questions right on papers the editor never saw (the 480
  questions of the 8 "hidden" papers of the group experiment, experiments/group/<model>/split.json)?
* Is that more than a random change of exactly the same shape and size would do?
* Does it damage the model (perplexity and looping, on the test split)?
* Does it survive compressing the model to 4-bit?

The steps, all run by `bench editor-eval --model gemma-4b-full --run <run>`:

1. describe     Load the change and write edit.json: which weights it touches (tensors, layers,
                rows or columns), how many numbers it changes, their share of the whole model,
                and per tensor how large the change is next to the weight itself (‖ΔW‖ / ‖W‖).
2. control      A random change touching exactly the same entries (same tensors, same rows or
                columns, same nonzero pattern) with random Gaussian values, scaled to the same
                Frobenius norm per tensor (fixed seed). Any change moves some answers by chance;
                the control shows how many.
3. full         Load the full-precision model once. Score the edited model and the control, each
                applied in memory and removed exactly afterwards, on: the hidden papers (480), the
                practice papers the editor worked on (drill 720 + validation 150, for reference),
                the test split (180, with the damage checks) and gemma_extra (120, Gemma only).
                The unedited model's runs are taken from experiments/group/<model>/baseline/ (same
                questions, same backend); only missing ones are scored, into
                experiments/editor/_baseline/<model>/.
4. quantized    Write the edited model to disk (work/editor/<run>/edited/), compress it to Q4_K_M
                exactly like the 4-bit baseline, score hidden, test (with damage checks) and
                gemma_extra, then delete the temporary files. The unedited 4-bit runs come from the
                group experiment (q4_hidden) and Phase 1 (gemma-4b-q4 on test and gemma_extra);
                missing ones are scored. The control is compressed too only with --with-control-q4.
5. collect      results.json and results.md: the editor's explanation and edit size, the practice
                score change (what the editor optimised), the hidden-paper numbers, the decision
                rule on the test split, gemma_extra, 4-bit survival, a per-section breakdown and a
                verdict in plain words.

The edit format
---------------
An edit is {weight name: change}, with names as in the loaded model (e.g.
`model.language_model.layers.23.mlp.down_proj.weight`). A change is a tensor of the weight's shape
(dense ΔW), or {"rows" | "cols": indices, "values": tensor}: values added to those whole rows (or
columns). bench.editor.workbench writes it with `save_edit` (keys `values:<name>`, `rows:<name>`,
`cols:<name>` and a `forms` entry in the file's metadata) and reads it with `load_edit`; this
module reads final_edit.safetensors with that `load_edit`. If the workbench cannot be imported or
fails, `read_edit_file` reads the same layout itself (and also `<name>`, `<name>.rows`,
`<name>.values` style keys). The control is written in the workbench's layout.

The change is added in float32 and the result rounded to the weight's own type (bfloat16), both
in memory and in the merged files; the same function does it in both places (the same arithmetic
as the workbench's `_add_delta`), so the 4-bit model is built from exactly the numbers that were
scored at full precision. Names on disk can differ from the names in the loaded model (Gemma 3
keeps `language_model.model.layers...` on disk): the run's editor_runs/<run>/parameters.json (the
workbench's table from model names to stored tensors) says where each one is; without it, a stored
name with the same parts in another order is taken, as in bench.edit.

How to read the results
-----------------------
* fixed:  of the questions the unedited model got wrong, the ones the edited model gets right.
* broken: of the questions the unedited model got right, the ones the edited model gets wrong.
  Net = fixed minus broken. "Wrong" and "right" come from the unedited model's run on the same
  backend (the 4-bit numbers use the unedited 4-bit model).
* The practice numbers only say how well the editor fit the questions it could look at.
* The key numbers are on the hidden papers: the net change against the unedited model, and the
  edited model minus the control, each with a 95% paired bootstrap interval (bench.stats). Only
  an interval entirely above zero counts as a gain; the report never calls anything an
  improvement when its interval includes zero.
* The test split gets the usual decision rule (bench.report, config/decision.toml): headline
  score (LÄS+MEK) up with the interval above zero, better than the control, no damage.

Resuming
--------
Every step writes its files and is skipped when they exist and are complete (a run folder is
complete when run.json has `finished_at`). An unfinished run folder is deleted and redone. The
model is loaded only when there is work left, and every action prints one timestamped line.
edit.json records the checksum of final_edit.safetensors; if the file changes afterwards the
driver stops rather than mix results of two different edits.

Folder layout:

    experiments/editor/
      <run>/
        edit.json                          step 1: the description of the change
        edited/                            the editor's change
          applied_change.json              how much of it survived rounding to bfloat16
          hidden/ drill/ validation/ test/ gemma_extra/      run folders, full precision
          q4_hidden/ q4_test/ q4_gemma_extra/                run folders, 4-bit
        control/                           the random change
          control_edit.safetensors, control.json, applied_change.json, same run folders
        comparisons/<name>/                bench.report comparisons
        results.json / results.md          step 5
      _baseline/<model>/                   unedited runs the group experiment did not have
    work/editor/<run>/<condition>/         temporary merged model and GGUF, deleted after scoring

Dry models (backend random/oracle/broken) have no weights: applying a change does nothing, and
compression writes a placeholder file whose "4-bit model" is the dry model's 4-bit entry in
config/models.toml. The tests replace `applied` with a stand-in to run the whole flow without a GPU.

torch and safetensors are imported inside the functions, so `import bench.editor.evaluate` works
without them.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import io
import json
import math
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping

from bench import group_experiment as ge
from bench import paths, report
from bench.backends import make_backend
from bench.backends.base import Backend
from bench.experiment import (
    DEFAULT_QUANT, LazyBackend, Measurement, compact_comparison, log, pending,
    q4_model_for, read_json, rel, run_is_complete, stable_seed, write_json_atomic, write_text_atomic,
)
from bench.models import ModelSpec, get_model
from bench.records import LETTERS, SECTIONS, Answer
from bench.run import run_with_backend
from bench.stats import bootstrap_mean_difference

if TYPE_CHECKING:
    import torch

EDITED, CONTROL = "edited", "control"
CONDITIONS = (EDITED, CONTROL)
PRACTICE_SETS = ("drill", "validation")
EDIT_STEM = "final_edit"              # editor_runs/<run>/final_edit.safetensors and .json
CONTROL_STEM = "control_edit"
CONTROL_SEED = 1
SPARSE_KINDS = ("rows", "cols")
SEPARATORS = ("::", "/", "|", "@", ".")   # between a weight name and rows/cols/values in sparse keys
WORKBENCH_LOADERS = ("load_final_edit", "load_edit")
WORKBENCH_PREFIXES = ("values:", "rows:", "cols:")   # key layout of bench.editor.workbench.save_edit
PARAMETER_TABLE = "parameters.json"   # editor_runs/<run>/parameters.json: model name -> stored tensor
INDICES_SHOWN = 24                     # edit.json lists at most this many row or column indices per tensor
IN_MEMORY_NOTE = ("`describe` lists the weight files on disk; the change under `experiment.edit` was "
                  "added to the weights in memory for this run")


# ================================================================== paths

def editor_runs_root() -> Path:
    return paths.ROOT / "editor_runs"


def edit_source(run: str) -> Path:
    """The editor's final edit, without suffix (.safetensors and .json)."""
    return editor_runs_root() / run / EDIT_STEM


def editor_root() -> Path:
    return paths.ROOT / "experiments" / "editor"


def run_root(run: str) -> Path:
    return editor_root() / run


def condition_dir(run: str, condition: str) -> Path:
    return run_root(run) / condition


def own_baseline_dir(model: str) -> Path:
    return editor_root() / "_baseline" / model


def work_dir(run: str, condition: str) -> Path:
    return paths.ROOT / "work" / "editor" / run / condition


def control_marker(run: str) -> Path:
    return condition_dir(run, CONTROL) / "control.json"


def run_label(model: str, run: str, condition: str) -> str:
    return f"{model}+editor[{run}]" if condition == EDITED else f"{model}+editor[{run}:random]"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ================================================================== the edit

@dataclass
class TensorEdit:
    """The change to one weight: the whole of it (dense), or some of its rows or columns."""
    name: str                          # parameter name in the loaded model
    kind: str                          # "dense", "rows" or "cols"
    values: "torch.Tensor"             # float32: ΔW, or the change of the chosen rows / columns
    index: "torch.Tensor | None" = None   # int64 row or column indices (sparse only)


@dataclass
class EditorEdit:
    tensors: list[TensorEdit]
    kind: str = "editor"               # "editor" or "random"
    info: dict = field(default_factory=dict)   # final_edit.json (or control.json)
    loader: str = ""                   # which function read the file


def load_edit(stem: Path) -> EditorEdit:
    """Read `<stem>.safetensors` (and `<stem>.json` when present), with the workbench's loader if
    it has one, otherwise with `read_edit_file`."""
    stem = Path(stem)
    tensor_file = stem.with_suffix(".safetensors")
    if not tensor_file.exists():
        raise FileNotFoundError(f"no {rel(tensor_file)}")
    info = read_json(stem.with_suffix(".json")) or {}
    mapping, loader = None, ""
    workbench = optional_workbench()
    for name in WORKBENCH_LOADERS:
        function = getattr(workbench, name, None) if workbench is not None else None
        if not callable(function):
            continue
        try:
            loaded = function(tensor_file)
        except Exception as error:   # the workbench is someone else's code; fall back to our reader
            log("editor", f"WARNING: bench.editor.workbench.{name} could not read {rel(tensor_file)} "
                          f"({error}); using this module's reader")
            break
        if isinstance(loaded, Mapping):
            mapping, loader = loaded, f"bench.editor.workbench.{name}"
        break
    if mapping is None:
        mapping, loader = read_edit_file(tensor_file), "bench.editor.evaluate.read_edit_file"
    return EditorEdit(tensors=tensor_edits(mapping), kind="editor", info=info, loader=loader)


def optional_workbench() -> Any | None:
    """bench.editor.workbench when it can be imported (it is written separately and may need torch)."""
    try:
        return importlib.import_module("bench.editor.workbench")
    except ImportError:
        return None


def read_edit_file(path: Path) -> dict[str, Any]:
    """The safetensors file as {name: tensor} (dense) or {name: {"rows"|"cols": idx, "values": t}}.

    Reads the workbench's layout (`values:<name>`, `rows:<name>`, `cols:<name>`) and, for files
    written by hand, `<name>` / `<name>.rows` / `<name>.values` keys."""
    from safetensors.torch import load_file

    flat = load_file(str(path))
    if any(key.startswith(WORKBENCH_PREFIXES) for key in flat):
        parts: dict[str, dict] = {}
        for key, tensor in flat.items():
            part, _, name = key.partition(":")
            if part + ":" not in WORKBENCH_PREFIXES or not name:
                raise ValueError(f"{path}: unexpected key {key!r} in a values:/rows:/cols: edit file")
            parts.setdefault(name, {})[part] = tensor
        return {name: entry["values"] if set(entry) == {"values"} else entry for name, entry in parts.items()}
    out: dict[str, Any] = {}
    for key, tensor in flat.items():
        name, part = split_sparse_key(key)
        if part is None:
            out[name] = tensor
        else:
            entry = out.setdefault(name, {})
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: {name!r} is stored both whole and as rows/columns")
            entry[part] = tensor
    return out


def split_sparse_key(key: str) -> tuple[str, str | None]:
    """'w.rows' -> ('w', 'rows'); 'w::values' -> ('w', 'values'); 'w.weight' -> ('w.weight', None)."""
    for part in (*SPARSE_KINDS, "values"):
        for separator in SEPARATORS:
            if key.endswith(separator + part):
                return key[: -len(separator + part)], part
    return key, None


def tensor_edits(mapping: Mapping[str, Any]) -> list[TensorEdit]:
    """Check the edit and turn it into TensorEdits (float32 values, int64 indices), sorted by name."""
    import torch

    edits = []
    for name in sorted(mapping):
        value = mapping[name]
        if isinstance(value, torch.Tensor):
            edits.append(TensorEdit(name, "dense", value.detach().to(torch.float32).cpu()))
            continue
        if not isinstance(value, Mapping):
            raise ValueError(f"edit of {name!r}: expected a tensor or {{rows|cols, values}}, got {type(value).__name__}")
        kinds = [kind for kind in SPARSE_KINDS if kind in value]
        if len(kinds) != 1 or "values" not in value:
            raise ValueError(f"edit of {name!r}: a sparse entry needs exactly one of rows/cols, and values; "
                             f"found {sorted(value)}")
        kind = kinds[0]
        index = torch.as_tensor(value[kind]).detach().to(torch.int64).cpu().reshape(-1)
        values = torch.as_tensor(value["values"]).detach().to(torch.float32).cpu()
        if index.unique().numel() != index.numel():
            raise ValueError(f"edit of {name!r}: repeated {kind} indices")
        if kind == "rows" and (values.dim() < 1 or values.shape[0] != index.numel()):
            raise ValueError(f"edit of {name!r}: {index.numel()} rows but values of shape {tuple(values.shape)}")
        if kind == "cols" and (values.dim() != 2 or values.shape[1] != index.numel()):
            raise ValueError(f"edit of {name!r}: {index.numel()} columns but values of shape {tuple(values.shape)}")
        edits.append(TensorEdit(name, kind, values, index))
    if not edits:
        raise ValueError("the edit changes no weights")
    return edits


def save_edit(edit: EditorEdit, stem: Path) -> Path:
    """Write the edit in the workbench's layout (used for the control), so either reader can open it."""
    from safetensors.torch import save_file

    flat, forms = {}, {}
    for part in edit.tensors:
        flat[f"values:{part.name}"] = part.values.contiguous()
        forms[part.name] = part.kind
        if part.kind != "dense":
            flat[f"{part.kind}:{part.name}"] = part.index.contiguous()
    path = Path(stem).with_suffix(".safetensors")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    save_file(flat, str(temporary), metadata={"forms": json.dumps(forms)})
    temporary.replace(path)
    return path


# ================================================================== the random control

def random_control(edit: EditorEdit, seed: int = CONTROL_SEED) -> EditorEdit:
    """Same tensors, same rows/columns, same nonzero pattern; Gaussian values scaled to the same
    Frobenius norm per tensor. Each tensor gets its own generator, seeded from `seed` and its name,
    so the control does not depend on the order of the tensors."""
    import torch

    tensors = []
    for part in edit.tensors:
        generator = torch.Generator().manual_seed(stable_seed(f"control|{seed}|{part.name}"))
        noise = torch.randn(part.values.shape, generator=generator, dtype=torch.float32)
        noise = noise * (part.values != 0)
        target, size = float(torch.linalg.norm(part.values)), float(torch.linalg.norm(noise))
        values = noise * (target / size) if size > 0 else torch.zeros_like(part.values)
        index = part.index.clone() if part.index is not None else None
        tensors.append(TensorEdit(part.name, part.kind, values, index))
    return EditorEdit(tensors=tensors, kind="random", info={"seed": seed}, loader="random_control")


# ================================================================== applying a change

def _added(original: "torch.Tensor", values: "torch.Tensor") -> "torch.Tensor":
    """original + values, computed in float32 and rounded back to the original's type. The one place
    the arithmetic happens: in memory (`applied`) and in files (`save_merged_model`) alike."""
    import torch

    return (original.to(torch.float32) + values.to(device=original.device, dtype=torch.float32)).to(original.dtype)


def _read_slice(weight: "torch.Tensor", part: TensorEdit, index: "torch.Tensor | None") -> "torch.Tensor":
    if part.kind == "dense":
        return weight.clone()
    return weight[index].clone() if part.kind == "rows" else weight[:, index].clone()


def _write_slice(weight: "torch.Tensor", part: TensorEdit, index: "torch.Tensor | None", new: "torch.Tensor") -> None:
    if part.kind == "dense":
        weight.copy_(new)
    elif part.kind == "rows":
        weight[index] = new
    else:
        weight[:, index] = new


def _check_shape(weight_shape: tuple, part: TensorEdit) -> None:
    shape = tuple(weight_shape)
    if part.kind == "dense":
        expected = shape
    elif part.kind == "rows":
        expected = (part.index.numel(), *shape[1:])
    else:
        expected = (shape[0], part.index.numel()) if len(shape) == 2 else None
    if expected is None or tuple(part.values.shape) != expected:
        raise ValueError(f"the {part.kind} change of {part.name} has shape {tuple(part.values.shape)}, "
                         f"which does not fit a weight of shape {shape}")
    if part.index is not None and part.index.numel():
        size = shape[0] if part.kind == "rows" else shape[1]
        if int(part.index.min()) < 0 or int(part.index.max()) >= size:
            raise ValueError(f"the {part.kind} indices of {part.name} go outside 0..{size - 1}")


def _landed(original: "torch.Tensor", new: "torch.Tensor", values: "torch.Tensor") -> dict:
    """How much of the intended change survived rounding to the weight's type."""
    import torch

    intended = float(torch.linalg.norm(values))
    realized = float(torch.linalg.norm(new.to(torch.float32) - original.to(torch.float32)))
    return {"intended_norm": intended, "realized_norm": realized,
            "realized_fraction": realized / intended if intended > 0 else 0.0,
            "entries_changed": int((new != original).sum())}


@contextmanager
def applied(backend: Backend, edit: EditorEdit | None) -> Iterator[list[dict]]:
    """Run a block with the change added to the backend's weights; the exact original numbers are
    put back afterwards (from copies of the touched entries, not by subtracting), also on errors.

    Yields, per tensor, how much of the change landed. A dry backend has no weights: nothing is
    changed and the yielded list says so."""
    if edit is None:
        yield []
        return
    model = getattr(backend, "model", None)
    if model is None:
        yield [{"note": "dry backend without weights: nothing was applied"}]
        return
    import torch

    saved: list[tuple[torch.Tensor, TensorEdit, Any, torch.Tensor]] = []
    landed = []
    try:
        with torch.no_grad():
            for part in edit.tensors:
                weight = model.get_parameter(part.name).data
                _check_shape(weight.shape, part)
                index = part.index.to(weight.device) if part.index is not None else None
                original = _read_slice(weight, part, index)
                new = _added(original, part.values)
                saved.append((weight, part, index, original))
                _write_slice(weight, part, index, new)
                landed.append({"name": part.name, **_landed(original, new, part.values)})
        yield landed
    finally:
        with torch.no_grad():
            for weight, part, index, original in reversed(saved):
                _write_slice(weight, part, index, original)


# ================================================================== weights on disk

def parameter_table(run: str) -> dict[str, dict] | None:
    """The workbench's table {model name: {"file", "key", "shape", "dtype"}} of this editor run, if any."""
    return read_json(editor_runs_root() / run / PARAMETER_TABLE)


def stored_key(keys: Iterable[str], param_name: str, table: Mapping[str, dict] | None = None) -> str:
    """The name under which a weight is stored in the model files. The workbench's table says so when
    given; else the name itself; else a stored name made of the same dot-separated parts in another
    order (Gemma 3: `model.language_model.layers.5...` in memory, `language_model.model.layers.5...`
    on disk); else the only stored name ending in the same `layers.<n>....` tail."""
    keys = list(keys)
    if table is not None and param_name in table and table[param_name]["key"] in keys:
        return table[param_name]["key"]
    if param_name in keys:
        return param_name
    parts = sorted(param_name.split("."))
    same = [key for key in keys if sorted(key.split(".")) == parts]
    if len(same) == 1:
        return same[0]
    if "layers." in param_name:
        tail = "layers." + param_name.split("layers.")[-1]
        ending = [key for key in keys if key == tail or key.endswith("." + tail)]
        if len(ending) == 1:
            return ending[0]
    raise ValueError(f"cannot find {param_name!r} among the stored weights (same parts: {same or 'none'})")


@dataclass
class StoredWeights:
    shards: list[Path]
    where: dict[str, Path]            # stored name -> shard
    shapes: dict[str, tuple]          # stored name -> shape

    @property
    def total(self) -> int:
        return sum(math.prod(shape) for shape in self.shapes.values())


def stored_weights(model_dir: Path) -> StoredWeights:
    from safetensors import safe_open

    shards = sorted(Path(model_dir).glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    where, shapes = {}, {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                where[key] = shard
                shapes[key] = tuple(f.get_slice(key).get_shape())
    return StoredWeights(shards, where, shapes)


def _frobenius(tensor: "torch.Tensor", rows_per_chunk: int = 4096) -> float:
    """‖tensor‖ in float32 chunks, so a large embedding matrix needs no full float32 copy."""
    import torch

    if tensor.dim() < 2:
        return float(torch.linalg.norm(tensor.to(torch.float32)))
    total = 0.0
    for start in range(0, tensor.shape[0], rows_per_chunk):
        total += float(tensor[start:start + rows_per_chunk].to(torch.float32).pow(2).sum())
    return math.sqrt(total)


def save_merged_model(src_dir: Path, edit: EditorEdit, out_dir: Path, table: Mapping[str, dict] | None = None) -> Path:
    """Write a loadable Hugging Face model folder that is `src_dir` with the change merged in.

    As in bench.group_edit.save_merged_model (and the workbench's save_edited_model): small files
    are copied, weight shards without a changed weight are symlinks to the originals, and the shards
    with one are rewritten under the same file names. `table` is the workbench's parameter table
    (see `stored_key`). The folder is built next to `out_dir` and renamed at the end. Returns `out_dir`."""
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")
    stored = stored_weights(src_dir)
    todo: dict[Path, list[tuple[str, TensorEdit]]] = {}
    for part in edit.tensors:
        key = stored_key(stored.where, part.name, table)
        _check_shape(stored.shapes[key], part)
        todo.setdefault(stored.where[key], []).append((key, part))

    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    for entry in sorted(src_dir.iterdir()):
        if entry.name.startswith("."):
            continue   # download caches, checksum sidecars and the like
        target = partial / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target)
        elif entry.suffix != ".safetensors":
            shutil.copy2(entry, target)
        elif entry not in todo:
            target.symlink_to(entry.resolve())

    rewritten = []
    for shard, entries in todo.items():
        with safe_open(str(shard), framework="pt") as f:
            file_metadata = f.metadata()
        tensors = load_file(str(shard))
        for key, part in entries:
            weight = tensors[key].clone()
            original = _read_slice(weight, part, part.index)
            new = _added(original, part.values)
            _write_slice(weight, part, part.index, new)
            tensors[key] = weight
            rewritten.append({"name": part.name, "stored_as": key, "file": shard.name,
                              **_landed(original, new, part.values)})
        save_file(tensors, str(partial / shard.name), metadata=file_metadata)

    provenance = {"kind": edit.kind, "source_model": str(src_dir), "rewritten": rewritten}
    (partial / "selma_editor_edit.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    partial.rename(out_dir)
    return out_dir


# ================================================================== step 1: describe

LAYER_PATTERN = re.compile(r"layers\.(\d+)\.(.+?)(?:\.weight|\.bias)?$")


def where_in_model(name: str) -> tuple[int | None, str]:
    """'model.language_model.layers.23.mlp.down_proj.weight' -> (23, 'mlp.down_proj')."""
    match = LAYER_PATTERN.search(name)
    if match:
        return int(match.group(1)), match.group(2)
    return None, name.removesuffix(".weight").removesuffix(".bias")


def describe_edit(edit: EditorEdit, model_dir: Path | None, table: Mapping[str, dict] | None = None) -> dict:
    """Which weights the change touches and how large it is. With the model files (`model_dir`) also
    each weight's shape and norm, how many entries really change once rounded to the stored type,
    and the share of the whole model; without them those are None.

    weights_touched  nonzero entries of the change (what the workbench calls the same)
    weights_changed  entries whose stored value differs after the float32 add and the rounding"""
    import torch

    stored = stored_weights(model_dir) if model_dir is not None else None
    rows = []
    for part in edit.tensors:
        layer, module = where_in_model(part.name)
        key = stored_key(stored.where, part.name, table) if stored else None
        shape = stored.shapes[key] if stored else (tuple(part.values.shape) if part.kind == "dense" else None)
        if shape is not None:
            _check_shape(shape, part)
        touched = int(torch.count_nonzero(part.values))
        delta_norm = float(torch.linalg.norm(part.values))
        row: dict[str, Any] = {
            "name": part.name, "stored_as": key, "layer": layer, "module": module, "kind": part.kind,
            "shape": list(shape) if shape is not None else None,
            "weights_touched": touched,
            "weights_changed": None,
            "share_of_tensor": touched / math.prod(shape) if shape else None,
            "delta_norm": delta_norm,
        }
        if part.kind == "dense" and part.values.dim() == 2:
            nonzero = part.values != 0
            row["rows_touched"] = int(nonzero.any(dim=1).sum())
            row["cols_touched"] = int(nonzero.any(dim=0).sum())
        elif part.kind != "dense":
            indices = sorted(int(i) for i in part.index)
            row[f"{part.kind}_touched"] = len(indices)
            row[f"{part.kind}_indices"] = indices[:INDICES_SHOWN]
            row[f"{part.kind}_indices_truncated"] = len(indices) > INDICES_SHOWN
        if stored:
            from safetensors import safe_open

            with safe_open(str(stored.where[key]), framework="pt") as f:
                weight = f.get_tensor(key)
            weight_norm = _frobenius(weight)
            original = _read_slice(weight, part, part.index)
            part_norm = _frobenius(original) if part.kind != "dense" else weight_norm
            row.update({"weight_norm": weight_norm,
                        "weights_changed": int((_added(original, part.values) != original).sum()),
                        "relative_norm": delta_norm / weight_norm if weight_norm else None,
                        "relative_norm_of_touched_part": delta_norm / part_norm if part_norm else None})
            del weight, original
        else:
            row.update({"weight_norm": None, "relative_norm": None, "relative_norm_of_touched_part": None})
        rows.append(row)

    total = stored.total if stored else None
    touched = sum(row["weights_touched"] for row in rows)
    modules: dict[str, int] = {}
    for row in rows:
        modules[row["module"]] = modules.get(row["module"], 0) + 1
    return {
        "n_tensors": len(rows),
        "weights_touched": touched,
        "weights_changed": sum(row["weights_changed"] for row in rows) if stored else None,
        "model_weights": total,
        "share_of_model": touched / total if total else None,
        "layers": sorted({row["layer"] for row in rows if row["layer"] is not None}),
        "modules": modules,
        "kinds": sorted({row["kind"] for row in rows}),
        "tensors": rows,
    }


def describe(model: str, run: str) -> dict:
    """Step 1. Write edit.json once; afterwards check that the edit file has not changed."""
    source = edit_source(run)
    tensor_file = source.with_suffix(".safetensors")
    if not tensor_file.exists():
        raise FileNotFoundError(f"no {rel(tensor_file)}: has the editor run {run!r} finished?")
    out = run_root(run) / "edit.json"
    record = read_json(out)
    checksum = sha256_file(tensor_file)
    if record is not None:
        if record["source_sha256"] != checksum:
            raise ValueError(f"{rel(tensor_file)} changed since {rel(out)} was written; delete "
                             f"{rel(run_root(run))} to evaluate the new edit")
        if record["model"] != model:
            raise ValueError(f"{rel(run_root(run))} was evaluated with {record['model']}, not {model}")
        log(tag(run), f"describe: done already ({rel(out)})")
        return record
    spec = get_model(model)
    edit = load_edit(source)
    editor_model = edit.info.get("model")
    if editor_model is not None and editor_model != model:
        raise ValueError(f"the editor run {run!r} edited {editor_model}, not {model}; use --model {editor_model}")
    model_dir = None if ge.is_dry(spec) else spec.path
    description = describe_edit(edit, model_dir, parameter_table(run))
    record = {
        "run": run, "model": model,
        "created_at": now(),
        "source": rel(tensor_file), "source_sha256": checksum, "loader": edit.loader,
        "editor": edit.info,
        "description": description,
        "model_weights_note": ("model_weights counts every number stored in the model files (vision "
                               "tower included)") if model_dir is not None else "dry model: no weight files",
    }
    write_json_atomic(out, record)
    share = description["share_of_model"]
    log(tag(run), f"describe: {description['weights_touched']:,} weights in {description['n_tensors']} tensors"
                  + (f" ({100 * share:.4f}% of the model)" if share is not None else "") + f" -> {rel(out)}")
    return record


def tag(run: str) -> str:
    return f"editor:{run}"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ================================================================== step 2: control

def make_control(run: str) -> None:
    """The random change with the same pattern and norms. Made once; control.json is written last."""
    marker = control_marker(run)
    if marker.exists():
        log(tag(run), f"control: done already ({rel(marker)})")
        return
    edit = load_edit(edit_source(run))
    control = random_control(edit, CONTROL_SEED)
    path = save_edit(control, condition_dir(run, CONTROL) / CONTROL_STEM)
    norms = [{"name": e.name, "edit_norm": float(_norm(e.values)), "control_norm": float(_norm(c.values))}
             for e, c in zip(edit.tensors, control.tensors)]
    write_json_atomic(marker, {"seed": CONTROL_SEED, "control_for": rel(edit_source(run).with_suffix(".safetensors")),
                               "file": rel(path), "rule": "same tensors, same rows/columns and nonzero pattern; "
                               "Gaussian values scaled to the same Frobenius norm per tensor", "norms": norms})
    log(tag(run), f"control: random change with seed {CONTROL_SEED} -> {rel(path)}")


def _norm(tensor: "torch.Tensor") -> float:
    import torch

    return float(torch.linalg.norm(tensor))


def load_condition_edit(run: str, condition: str) -> EditorEdit:
    if condition == EDITED:
        return load_edit(edit_source(run))
    if not control_marker(run).exists():
        raise FileNotFoundError(f"no {rel(control_marker(run))}; run `bench editor-eval --run {run}` first")
    control = load_edit(condition_dir(run, CONTROL) / CONTROL_STEM)
    control.kind, control.info = "random", read_json(control_marker(run)) or {}
    return control


def edit_brief(run: str, edit: EditorEdit) -> dict:
    """The change in a few numbers, for run.json."""
    return {"run": run, "kind": edit.kind, "loader": edit.loader, "n_tensors": len(edit.tensors),
            "tensors": [{"name": t.name, "kind": t.kind, "values_shape": list(t.values.shape)} for t in edit.tensors]}


# ================================================================== scoring

def score(backend: Backend, label: str, run_dir: Path, measurement: Measurement, tokenizer_path: str | None,
          experiment_info: dict, who: str) -> None:
    """One run folder. An unfinished folder from an interrupted run is removed first."""
    if run_dir.exists():
        log(who, f"removing unfinished run folder {rel(run_dir)}")
        shutil.rmtree(run_dir)
    checks = ", with damage checks" if measurement.damage else ""
    log(who, f"scoring {label} on {measurement.name}: {len(measurement.questions)} questions{checks}")
    run_with_backend(backend, label, measurement.split, measurement.questions, tokenizer_path,
                     skip_damage=not measurement.damage,
                     extra_info={"experiment": {**experiment_info, "question_set": measurement.name}},
                     run_dir=run_dir)
    summary = read_json(run_dir / "summary.json")
    log(who, f"finished {rel(run_dir)}: accuracy {summary['overall']['accuracy']:.1%}")


def measure(get_backend: Callable[[], Backend], folder: Path, label: str, measurements: list[Measurement],
            tokenizer_path: str | None, who: str, info: dict, edit: EditorEdit | None = None,
            run: str | None = None) -> None:
    """Produce the missing run folders, with `edit` added to the weights in memory while they run."""
    todo = pending(folder, measurements)
    if not todo:
        log(who, f"{label}: {len(measurements)} runs done already")
        return
    backend = get_backend()
    with applied(backend, edit) as landed:
        experiment_info = dict(info)
        if edit is not None:
            experiment_info.update({"edit": edit_brief(run or "", edit), "applied_change": landed,
                                    "note": IN_MEMORY_NOTE})
            write_json_atomic(folder / "applied_change.json", landed)
        for measurement in todo:
            score(backend, label, folder / measurement.name, measurement, tokenizer_path, experiment_info, who)


def baseline_candidates(model: str, measurement: Measurement) -> list[Path]:
    """Where an unedited run of this question set may already be, best first."""
    out = []
    if measurement.name in ("q4_test", "q4_gemma_extra"):
        phase1 = ge.q4_phase1_run(model, measurement.name.removeprefix("q4_"))
        if phase1 is not None:
            out.append(phase1)
    return out + [ge.condition_dir(model, ge.BASELINE) / measurement.name, own_baseline_dir(model) / measurement.name]


def baseline_run(model: str, measurement: Measurement) -> Path:
    """The first complete unedited run of this set; our own folder (maybe still to be made) otherwise."""
    for candidate in baseline_candidates(model, measurement):
        if run_is_complete(candidate, measurement.damage):
            return candidate
    return own_baseline_dir(model) / measurement.name


def missing_baselines(model: str, measurements: list[Measurement]) -> list[Measurement]:
    return [m for m in measurements if not run_is_complete(baseline_run(model, m), m.damage)]


def read_group_split(model: str) -> dict:
    split = read_json(ge.split_path(model))
    if split is None:
        raise FileNotFoundError(f"no {rel(ge.split_path(model))}; the hidden papers come from the group "
                                f"experiment: run `bench group split --model {model}` first")
    return split


def full_sets(model: str, split: dict) -> list[Measurement]:
    return list(ge.question_sets(model, split).values())


def q4_sets(model: str, split: dict) -> list[Measurement]:
    return list(ge.q4_question_sets(model, split).values())


# ================================================================== step 3: full precision

def full_precision(model: str, run: str, session: LazyBackend | None = None) -> None:
    """Step 3. Unedited runs that are missing, then the edit and the control, in one loaded model."""
    split = read_group_split(model)
    spec = get_model(model)
    tokenizer = ge.tokenizer_of(spec)
    measurements = full_sets(model, split)
    own_session = session is None
    session = session or LazyBackend(tag(run), lambda: make_backend(model), model)
    try:
        missing = missing_baselines(model, measurements)
        if missing:
            measure(session.get, own_baseline_dir(model), model, missing, tokenizer, tag(run),
                    info={"base_model": model, "condition": "baseline", "precision": "full"})
        else:
            log(tag(run), f"baseline: {len(measurements)} unedited runs found")
        for condition in CONDITIONS:
            folder = condition_dir(run, condition)
            if not pending(folder, measurements):
                log(tag(run), f"full precision {condition}: {len(measurements)} runs done already")
                continue
            edit = load_condition_edit(run, condition)
            measure(session.get, folder, run_label(model, run, condition), measurements, tokenizer, tag(run),
                    info={"base_model": model, "editor_run": run, "condition": condition, "precision": "full"},
                    edit=edit, run=run)
    finally:
        if own_session:
            session.close()
    log(tag(run), "full precision: finished")


# ================================================================== step 4: 4-bit

def quantized(model: str, run: str, with_control: bool = False) -> None:
    """Step 4. Missing unedited 4-bit runs, then each change merged, compressed and scored."""
    split = read_group_split(model)
    spec = get_model(model)
    q4_name = q4_model_for(model)
    q4_spec = get_model(q4_name)
    q4_tokenizer = ge.tokenizer_of(q4_spec)
    measurements = q4_sets(model, split)

    missing = missing_baselines(model, measurements)
    if missing:
        baseline = LazyBackend(tag(run), lambda: make_backend(q4_name), f"{q4_name} (unedited 4-bit)")
        try:
            measure(baseline.get, own_baseline_dir(model), q4_name, missing, q4_tokenizer, tag(run),
                    info={"base_model": model, "condition": "baseline", "precision": "q4"})
        finally:
            baseline.close()
    else:
        log(tag(run), f"4-bit baseline: {len(measurements)} unedited runs found")

    for condition in (CONDITIONS if with_control else (EDITED,)):
        quantize_one(model, run, spec, q4_spec, q4_tokenizer, condition, measurements)
    log(tag(run), "quantized: finished")


def quantize_one(model: str, run: str, spec: ModelSpec, q4_spec: ModelSpec, q4_tokenizer: str | None,
                 condition: str, measurements: list[Measurement]) -> None:
    folder = condition_dir(run, condition)
    label = run_label(q4_spec.name, run, condition)
    if not pending(folder, measurements):
        log(tag(run), f"{label}: {len(measurements)} runs done already")
        return
    edit = load_condition_edit(run, condition)
    work = work_dir(run, condition)
    quant = str(q4_spec.setting("compression", DEFAULT_QUANT))
    gguf = merged_gguf(run, spec, edit, work, quant)
    backend = LazyBackend(tag(run), lambda: ge.open_gguf(gguf, q4_spec), f"4-bit edited model {rel(gguf)}")
    try:
        # The change is in the file, so nothing is applied in memory here.
        measure(backend.get, folder, label, measurements, q4_tokenizer, tag(run),
                info={"base_model": model, "editor_run": run, "condition": condition, "precision": "q4",
                      "edit_in_file": edit_brief(run, edit), "gguf": rel(gguf), "quant": quant})
    finally:
        backend.close()
    shutil.rmtree(work)   # several GB; it can be rebuilt from the saved edit
    log(tag(run), f"deleted {rel(work)}")


def merged_gguf(run: str, spec: ModelSpec, edit: EditorEdit, work: Path, quant: str) -> Path:
    """Write the changed model to disk, compress it like the 4-bit baseline, delete the full-size copy."""
    gguf = work / f"merged-{quant}.gguf"
    if ge.gguf_done(gguf):
        log(tag(run), f"reusing finished {rel(gguf)}")
        return gguf
    hf_dir = work / "hf"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)   # left over from an interrupted run
    work.mkdir(parents=True, exist_ok=True)
    log(tag(run), f"saving the changed model to {rel(hf_dir)}")
    if ge.is_dry(spec):
        hf_dir.mkdir()
        (hf_dir / "DRY_PLACEHOLDER").write_text("dry model: no weights to merge\n", encoding="utf-8")
    else:
        if spec.path is None:
            raise ValueError(f"model {spec.name!r} has no path in config/models.toml")
        save_merged_model(spec.path, edit, hf_dir, parameter_table(run))
    log(tag(run), f"compressing to {quant}: {rel(gguf)}")
    ge.compress(spec, hf_dir, gguf, quant)
    shutil.rmtree(hf_dir)
    return gguf


# ================================================================== step 5: collect

def collect(model: str, run: str) -> Path:
    """Step 5. results.json and results.md from whatever runs exist. Cheap, so it always runs."""
    record = read_json(run_root(run) / "edit.json") or describe(model, run)
    split = read_group_split(model)
    decision = report.load_decision()
    results = {
        "run": run,
        "model": model,
        "created_at": now(),
        "source": record["source"],
        "source_sha256": record["source_sha256"],
        "editor": editor_facts(record.get("editor") or {}, run),
        "editor_record": record.get("editor"),
        "edit": record["description"],
        "control": read_json(control_marker(run)),
        "applied_change": {c: read_json(condition_dir(run, c) / "applied_change.json") for c in CONDITIONS},
        "confidence": decision["confidence"],
        "full": precision_results(model, run, split, "full", decision),
        "q4": precision_results(model, run, split, "q4", decision),
    }
    results["survival"] = survival(results)
    results["verdict"] = verdict(results)
    out = run_root(run) / "results.json"
    write_json_atomic(out, results)
    write_text_atomic(run_root(run) / "results.md", render_results_md(results))
    log(tag(run), f"collect: -> {rel(out)} and results.md")
    return out


def flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, Mapping):
        out: dict[str, Any] = {}
        for key, value in data.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: data}


def editor_facts(info: dict, run: str) -> dict:
    """The few facts results.md needs from final_edit.json, found by key name (the workbench's exact
    field names are not fixed here): explanation, forward-pass budget, passes used, edit format."""
    flat = flatten(info)

    def find(test: Callable[[str], bool], kinds: tuple = (int, float, str)) -> Any:
        for key, value in flat.items():
            last = key.lower().split(".")[-1]
            if test(last) and isinstance(value, kinds) and not isinstance(value, bool):
                return value
        return None

    budget = find(lambda k: "budget" in k, (int, float))
    guessed = False
    if budget is None:
        digits = re.findall(r"\d+", run)
        budget, guessed = (int(digits[0]), True) if digits else (None, False)
    size_record = info.get("size") if isinstance(info.get("size"), Mapping) else {}
    size = {k: v for k, v in size_record.items() if isinstance(v, (int, float, str))} or None
    return {
        "explanation": find(lambda k: "explanation" in k or "rationale" in k, (str,)),
        "budget": budget,
        "budget_from_run_name": guessed,
        "passes_used": find(lambda k: "pass" in k and ("used" in k or "spent" in k), (int, float)),
        "format": find(lambda k: "format" in k, (str,)),
        "size": size,       # the scalar entries of final_edit.json's "size" (the workbench's edit_size)
    }


def sets_of(model: str, precision: str) -> list[str]:
    sets = ["hidden", *PRACTICE_SETS, "test"] if precision == "full" else ["hidden", "test"]
    return sets + (["gemma_extra"] if ge.has_gemma_extra(model) else [])


def measurement_named(model: str, split: dict, precision: str, set_name: str) -> Measurement:
    sets = full_sets(model, split) if precision == "full" else q4_sets(model, split)
    name = ge.run_name(precision, set_name)
    return next(m for m in sets if m.name == name)


def precision_results(model: str, run: str, split: dict, precision: str, decision: dict) -> dict | None:
    """All numbers of one precision. None when neither change has a run at it."""
    sets = sets_of(model, precision)
    folder = {s: ge.run_name(precision, s) for s in sets}
    base_dirs = {s: baseline_run(model, measurement_named(model, split, precision, s)) for s in sets}
    before = {s: ge.complete_answers(base_dirs[s]) for s in sets}
    after = {c: {s: ge.complete_answers(condition_dir(run, c) / folder[s]) for s in sets} for c in CONDITIONS}
    if not any(answers for runs in after.values() for answers in runs.values()):
        return None

    out: dict[str, Any] = {"baselines": {s: rel(base_dirs[s]) if before[s] else None for s in sets}}
    hidden_ids = split["question_ids"]["hidden_wrong"] + split["question_ids"]["hidden_right"]
    if before["hidden"] is not None:
        out["hidden"] = flip_block(after, before["hidden"], "hidden", hidden_ids, decision)
        if precision == "full":
            out["hidden_check"] = split_agreement(split, before["hidden"])

    if precision == "full":
        practice_before = merged_answers([before[s] for s in PRACTICE_SETS])
        practice_after = {c: {"practice": merged_answers([after[c][s] for s in PRACTICE_SETS])} for c in CONDITIONS}
        if practice_before is not None:
            out["practice"] = flip_block(practice_after, practice_before, "practice", list(practice_before), decision)

    for s in ("test", "gemma_extra"):
        if s not in sets:
            continue
        edited, control = condition_dir(run, EDITED) / folder[s], condition_dir(run, CONTROL) / folder[s]
        out[s] = {
            "edited_vs_baseline": compare(run, f"{precision}_edited_vs_baseline_{s}", edited, base_dirs[s], control),
            "control_vs_baseline": compare(run, f"{precision}_control_vs_baseline_{s}", control, base_dirs[s], None),
            "edited_vs_control": compare(run, f"{precision}_edited_vs_control_{s}", edited, control, None),
        }
    return out


def merged_answers(runs: list[dict[str, Answer] | None]) -> dict[str, Answer] | None:
    if any(answers is None for answers in runs):
        return None
    merged: dict[str, Answer] = {}
    for answers in runs:
        merged.update(answers)
    return merged


def flip_block(after: dict, before: dict[str, Answer], set_name: str, ids: list[str], decision: dict) -> dict:
    """Fixed, broken and net for each change against the unedited model, and edited minus control."""
    block: dict[str, Any] = {}
    for condition in CONDITIONS:
        answers = after[condition].get(set_name)
        block[condition] = flips(answers, before, ids, decision) if answers is not None else None
        block[f"{condition}_by_section"] = by_section(answers, before, ids) if answers is not None else None
    edited, control = after[EDITED].get(set_name), after[CONTROL].get(set_name)
    if edited is not None and control is not None:
        wrong = [qid for qid in ids if not before[qid].is_correct]
        right = [qid for qid in ids if before[qid].is_correct]
        block["edited_vs_control"] = {"wrong_fixed": ge.paired(edited, control, wrong, decision),
                                      "right_kept": ge.paired(edited, control, right, decision),
                                      "net": ge.paired(edited, control, ids, decision),
                                      "prob_correct": paired_prob(edited, control, ids, decision)}
    else:
        block["edited_vs_control"] = None
    return block


def flips(after: dict[str, Answer], before: dict[str, Answer], ids: list[str], decision: dict) -> dict:
    wrong = [qid for qid in ids if not before[qid].is_correct]
    right = [qid for qid in ids if before[qid].is_correct]
    fixed = sum(after[qid].is_correct for qid in wrong)
    kept = sum(after[qid].is_correct for qid in right)
    a = [float(after[qid].is_correct) for qid in ids]
    b = [float(before[qid].is_correct) for qid in ids]
    change, low, high = bootstrap_mean_difference(a, b, confidence=decision["confidence"],
                                                  samples=decision["bootstrap_samples"])
    return {
        "n": len(ids), "accuracy_before": sum(b) / len(ids), "accuracy_after": sum(a) / len(ids),
        "n_wrong": len(wrong), "fixed": fixed, "fixed_share": ge.share(fixed, len(wrong)),
        "n_right": len(right), "kept": kept, "broken": len(right) - kept,
        "broken_share": ge.share(len(right) - kept, len(right)),
        "net_questions": fixed - (len(right) - kept),
        "net": {"accuracy_change": change, "ci": [low, high]},
        "prob_correct": paired_prob(after, before, ids, decision),
        "fixed_ids": [qid for qid in wrong if after[qid].is_correct],
        "broken_ids": [qid for qid in right if not after[qid].is_correct],
    }


def paired_prob(first: dict[str, Answer], second: dict[str, Answer], ids: list[str], decision: dict) -> dict:
    change, low, high = bootstrap_mean_difference([first[q].prob_correct for q in ids],
                                                  [second[q].prob_correct for q in ids],
                                                  confidence=decision["confidence"], samples=decision["bootstrap_samples"])
    return {"change": change, "ci": [low, high]}


def by_section(after: dict[str, Answer], before: dict[str, Answer], ids: list[str]) -> dict:
    rows = {}
    for section in SECTIONS:
        chosen = [qid for qid in ids if before[qid].section == section]
        wrong = [qid for qid in chosen if not before[qid].is_correct]
        right = [qid for qid in chosen if before[qid].is_correct]
        fixed = sum(after[qid].is_correct for qid in wrong)
        broken = sum(not after[qid].is_correct for qid in right)
        rows[section] = {"n": len(chosen), "n_wrong": len(wrong), "fixed": fixed, "n_right": len(right),
                         "broken": broken, "net_questions": fixed - broken,
                         "accuracy_before": ge.share(len(right), len(chosen)),
                         "accuracy_after": ge.share(len(right) - broken + fixed, len(chosen))}
    return rows


def split_agreement(split: dict, before: dict[str, Answer]) -> dict:
    """How many hidden questions the unedited run judges differently from split.json (should be 0)."""
    ids = split["question_ids"]
    disagree = [qid for qid in ids["hidden_wrong"] if before[qid].is_correct]
    disagree += [qid for qid in ids["hidden_right"] if not before[qid].is_correct]
    return {"split_from": split.get("baseline_run"), "disagreements": len(disagree), "question_ids": disagree}


def compare(run: str, name: str, edited: Path, baseline: Path, control: Path | None) -> dict | None:
    """bench.report.compare_runs on two (or three) run folders; None while a run is missing."""
    if not run_is_complete(edited) or not run_is_complete(baseline):
        return None
    if control is not None and not run_is_complete(control):
        control = None
    out_root = run_root(run) / "comparisons" / name
    with contextlib.redirect_stdout(io.StringIO()):   # compare_runs prints the report; it is in comparison.md
        result = report.compare_runs(edited, baseline, control=control, out_root=out_root)
    return compact_comparison(result, out_root / f"{result['edited']}__vs__{result['baseline']}")


def survival(results: dict) -> dict | None:
    """Does the full-precision effect on the hidden papers carry over to the 4-bit model?"""
    full = ((results.get("full") or {}).get("hidden") or {}).get(EDITED)
    q4 = ((results.get("q4") or {}).get("hidden") or {}).get(EDITED)
    if full is None or q4 is None:
        return None
    full_gain = ge.above_zero(full["net"]["ci"])
    q4_gain = ge.above_zero(q4["net"]["ci"])
    status = ("survives" if full_gain and q4_gain else
              "lost" if full_gain else
              "no_full_precision_gain")
    return {"status": status, "full_net": full["net"], "q4_net": q4["net"],
            "full_net_questions": full["net_questions"], "q4_net_questions": q4["net_questions"],
            "fixed_at_full_also_fixed_at_q4": len(set(full["fixed_ids"]) & set(q4["fixed_ids"])),
            "n_fixed_at_full": len(full["fixed_ids"])}


# ================================================================== the verdict in plain words

def classify(ci: list[float] | None) -> str | None:
    if ci is None:
        return None
    return "above" if ge.above_zero(ci) else "below" if ge.below_zero(ci) else "zero"


def verdict(results: dict) -> dict:
    """Fixed rules, so the same numbers always give the same words. Nothing is called a gain or an
    improvement unless its confidence interval lies entirely above zero.

    Overall label:
      improvement             hidden net vs unedited above zero, AND hidden edited minus random above
                              zero, AND the test split's damage rules (3a, 3b) passed
      harm                    hidden net vs unedited below zero, or a damage rule failed
      no evidence             everything else (including "not measured yet")
    """
    full = results.get("full") or {}
    first = [edit_sentence(results)]
    practice = full.get("practice") or {}
    if practice.get(EDITED):
        first.append(practice_sentence(practice))
    hidden = full.get("hidden") or {}
    if hidden.get(EDITED):
        second = hidden_sentences(hidden, f"On the hidden papers ({hidden[EDITED]['n']} questions the editor "
                                          "never saw)")
    else:
        second = ["The hidden papers have not been scored yet."]
    third: list[str] = []
    test = (full.get("test") or {}).get("edited_vs_baseline")
    damage_ok = None
    if test is not None:
        third.append(test_sentence(test))
        damage_ok, damage_text = damage_sentence(test)
        third.append(damage_text)
    extra = (full.get("gemma_extra") or {}).get("edited_vs_baseline")
    if extra is not None:
        overall = extra["overall"]
        third.append(f"On gemma_extra (indicative only) accuracy changes by {ge.pp(overall['accuracy_change'])} "
                     f"({ge.ci_text(overall['accuracy_ci'])}); " + interval_words(overall["accuracy_ci"]) + ".")
    fourth = [q4_sentence(results)]

    net_class = classify((hidden.get(EDITED) or {}).get("net", {}).get("ci"))
    versus_class = classify(((hidden.get("edited_vs_control") or {}).get("net") or {}).get("ci"))
    if net_class == "below" or damage_ok is False:
        label = "harm"
        closing = ("Overall: the edit does harm" + (" (it damages the model)" if damage_ok is False else "")
                   + "; it is not an improvement.")
    elif net_class == "above" and versus_class == "above" and damage_ok is True:
        label = "improvement"
        closing = ("Overall: an improvement on unseen papers, larger than a random edit of the same shape, "
                   "with no damage.")
    else:
        label = "no evidence"
        closing = ("Overall: no evidence that the edit makes the model better at Swedish beyond the questions "
                   "the editor worked on.")
    paragraphs = [" ".join(group) for group in (first, second, third, fourth, [closing]) if group]
    return {"label": label, "paragraph": "\n\n".join(paragraphs)}


def interval_words(ci: list[float]) -> str:
    kind = classify(ci)
    if kind == "above":
        return "the interval lies above zero, a real gain"
    if kind == "below":
        return "the interval lies below zero, a real loss"
    return "the interval includes zero, so it cannot be told apart from no change"


def edit_sentence(results: dict) -> str:
    edit, facts = results["edit"], results["editor"]
    budget = ""
    if facts.get("budget") is not None:
        budget = f" with a budget of {facts['budget']} forward passes"
        if facts.get("passes_used") is not None:
            budget += f" (used {facts['passes_used']})"
    share = edit.get("share_of_model")
    where = f"layers {', '.join(map(str, edit['layers']))}" if edit.get("layers") else "no decoder layer"
    return (f"The editor{budget} changed {edit['weights_touched']:,} weights in {edit['n_tensors']} tensors"
            + (f" ({100 * share:.4f}% of the model)" if share is not None else "") + f", in {where}.")


def practice_sentence(practice: dict) -> str:
    edited = practice[EDITED]
    text = (f"On the practice papers ({edited['n']} questions the editor could use) accuracy goes from "
            f"{ge.pct(edited['accuracy_before'])} to {ge.pct(edited['accuracy_after'])}, "
            f"{ge.pp(edited['net']['accuracy_change'])} ({ge.ci_text(edited['net']['ci'])})")
    control = practice.get(CONTROL)
    if control is not None:
        text += f"; the random edit: {ge.pp(control['net']['accuracy_change'])}"
    return text + ". This is what the editor optimised, so it only shows how well the edit fits those questions."


def hidden_sentences(hidden: dict, name: str) -> list[str]:
    edited = hidden[EDITED]
    net = edited["net"]
    out = [f"{name} the edit fixes {edited['fixed']} of {edited['n_wrong']} wrong answers and breaks "
           f"{edited['broken']} of {edited['n_right']} right ones: net {edited['net_questions']:+d} questions, "
           f"{ge.pp(net['accuracy_change'])} accuracy ({ge.ci_text(net['ci'])})."]
    kind = classify(net["ci"])
    out.append({"above": "The interval lies above zero: a real net gain over the unedited model.",
                "below": "The interval lies below zero: a real net loss against the unedited model.",
                "zero": "The interval includes zero, so the net change cannot be told apart from no change."}[kind])
    versus = (hidden.get("edited_vs_control") or {}).get("net")
    control = hidden.get(CONTROL)
    if versus is None or control is None:
        out.append("There is no random-edit run to compare with, so it is not known whether that is more than "
                   "any edit of this shape would do.")
        return out
    compared = (f"The random edit of the same shape and size nets {control['net_questions']:+d} questions; edited "
                f"minus random is {ge.pp(versus['change'])} ({ge.ci_text(versus['ci'])}).")
    out.append(compared + {
        "above": " The interval lies above zero: the edit does better than a random edit of the same shape.",
        "below": " The interval lies below zero: the edit does worse than a random edit of the same shape.",
        "zero": " The interval includes zero, so the edit does no better than a random edit might by chance.",
    }[classify(versus["ci"])])
    return out


def test_sentence(test: dict) -> str:
    headline = test["headline"]
    interval = f"95% CI {headline['metric_ci'][0]:+.4f} to {headline['metric_ci'][1]:+.4f}"
    if test["verdict"] == "IMPROVEMENT":
        return (f"On the test split the decision rule says IMPROVEMENT: the headline score (LÄS+MEK, mean "
                f"probability on the correct answer) changes by {headline['metric_change']:+.4f} ({interval}), it "
                "beats the random edit and there is no damage.")
    failed = [rule for rule, entry in test["rules"].items() if not entry["passed"]]
    return (f"On the test split the decision rule says NO IMPROVEMENT (headline change "
            f"{headline['metric_change']:+.4f}, {interval}; rules not passed: {', '.join(failed)}).")


def damage_sentence(test: dict) -> tuple[bool | None, str]:
    damage = test.get("damage")
    if damage is None:
        return None, "The damage checks are missing for the test split."
    rules = test["rules"]
    ok = rules["3a_perplexity"]["passed"] and rules["3b_loop_rate"]["passed"]
    text = (f"Damage checks: perplexity {damage['perplexity']['relative_change']:+.2%}, loop rate "
            f"{damage['loop_rate_percent']['change_points']:+.1f} points: "
            + ("within the limits." if ok else "outside the limits, so the edit damages the model."))
    return ok, text


def q4_sentence(results: dict) -> str:
    q4 = results.get("q4")
    if q4 is None or not (q4.get("hidden") or {}).get(EDITED):
        return "The 4-bit version has not been measured yet."
    edited = q4["hidden"][EDITED]
    text = (f"After compression to 4-bit, against the unedited 4-bit model, the hidden-paper net change is "
            f"{edited['net_questions']:+d} questions, {ge.pp(edited['net']['accuracy_change'])} "
            f"({ge.ci_text(edited['net']['ci'])}); " + interval_words(edited["net"]["ci"]) + ".")
    status = (results.get("survival") or {}).get("status")
    if status == "survives":
        text += " The full-precision gain survives compression."
    elif status == "lost":
        text += " The full-precision gain does not survive compression."
    elif status == "no_full_precision_gain":
        text += " There was no full-precision gain to carry over."
    return text


# ================================================================== results.md

def render_results_md(results: dict) -> str:
    run, edit, facts = results["run"], results["edit"], results["editor"]
    lines = [
        f"# Editor edit: {run}",
        "",
        f"One weight edit of `{results['model']}` made by an AI editor (`{results['source']}`), scored on papers the "
        "editor never saw (the hidden papers of the group experiment), against a random edit touching exactly the "
        "same weights with the same size. See `src/bench/editor/evaluate.py` for the method.",
        "",
        "## Verdict",
        "",
        f"**Overall label: {results['verdict']['label']}**",
        "",
        results["verdict"]["paragraph"],
        "",
        "## The edit",
        "",
        f"* Budget: {facts['budget'] if facts['budget'] is not None else 'not recorded'} forward passes"
        + (" (from the run name)" if facts.get("budget_from_run_name") else "")
        + (f"; used {facts['passes_used']}" if facts.get("passes_used") is not None else ""),
        f"* Format: {facts['format'] or ', '.join(edit['kinds'])}",
        f"* Weights touched (nonzero in the change): {edit['weights_touched']:,} in {edit['n_tensors']} tensors"
        + (f", {100 * edit['share_of_model']:.4f}% of the model's {edit['model_weights']:,} weights"
           if edit.get("share_of_model") is not None else "")
        + (f"; {edit['weights_changed']:,} of them still differ after rounding to bfloat16"
           if edit.get("weights_changed") is not None else ""),
        f"* Layers: {', '.join(map(str, edit['layers'])) or 'none'}; modules: "
        + ", ".join(f"{name} ({count})" for name, count in edit["modules"].items()),
    ]
    if facts.get("size"):
        lines.append(f"* Size as reported by the editor: `{json.dumps(facts['size'], ensure_ascii=False)}`")
    lines += ["", "The editor's explanation:", ""]
    explanation = facts.get("explanation") or "(none recorded)"
    lines += [f"> {line}" if line.strip() else ">" for line in explanation.splitlines()]
    landed = {c: {row.get("name"): row for row in (results["applied_change"].get(c) or []) if isinstance(row, dict)}
              for c in CONDITIONS}
    lines += ["", "| Tensor | Layer | Kind | Rows / columns touched | Weights touched | Share of tensor | "
                  "‖ΔW‖ / ‖W‖ | Norm landed in bf16 (edit / random) |", "|---|---|---|---|---|---|---|---|"]
    for row in edit["tensors"]:
        lines.append(f"| `{row['module']}` | {row['layer'] if row['layer'] is not None else '—'} | {row['kind']} | "
                     f"{touched_text(row)} | {row['weights_touched']:,} | {ge.pct(row['share_of_tensor'])} | "
                     f"{number(row['relative_norm'], 4)} | {landed_text(landed, row['name'])} |")
    control = results.get("control")
    lines += ["", f"Random control: seed {control['seed']}; {control['rule']}." if control else
              "Random control: not made yet."]

    for precision, title in (("full", "Full precision"), ("q4", "4-bit (Q4_K_M)")):
        numbers = results.get(precision)
        lines += ["", f"## {title}", ""]
        if numbers is None:
            lines.append("Not measured yet.")
            continue
        lines += precision_md(numbers, precision)
    lines += ["", "## Unedited runs used", ""]
    for precision in ("full", "q4"):
        for name, path in ((results.get(precision) or {}).get("baselines") or {}).items():
            lines.append(f"* {precision} {name}: `{path or 'missing'}`")
    lines.append("")
    return "\n".join(lines)


def touched_text(row: dict) -> str:
    if row["kind"] == "dense":
        if "rows_touched" in row:
            return f"{row['rows_touched']} rows × {row['cols_touched']} columns (nonzero)"
        return "all"
    kind = row["kind"]
    indices = ", ".join(map(str, row[f"{kind}_indices"])) + (", …" if row[f"{kind}_indices_truncated"] else "")
    return f"{row[f'{kind}_touched']} {'rows' if kind == 'rows' else 'columns'}: {indices}"


def landed_text(landed: dict, name: str) -> str:
    cells = []
    for condition in CONDITIONS:
        row = landed[condition].get(name)
        cells.append(ge.pct(row["realized_fraction"]) if row else "—")
    return " / ".join(cells)


def number(value: float | None, digits: int) -> str:
    return "n/a" if value is None else f"{value:.{digits}g}"


def precision_md(numbers: dict, precision: str) -> list[str]:
    lines: list[str] = []
    check = numbers.get("hidden_check")
    if check is not None:
        lines += [f"The unedited hidden-paper run disagrees with split.json on {check['disagreements']} questions.", ""]
    blocks = [("practice", "Practice papers (drill + validation; what the editor optimised)"),
              ("hidden", "Hidden papers (never seen by the editor)")]
    for key, title in blocks:
        block = numbers.get(key)
        if not block:
            continue
        lines += [f"### {title}", "",
                  "| Edit | Questions | Accuracy before → after | Fixed | Broken | Net (accuracy, CI) | "
                  "Mean prob. correct change (CI) |", "|---|---|---|---|---|---|---|"]
        for condition in CONDITIONS:
            entry = block.get(condition)
            if entry is None:
                continue
            lines.append(f"| {condition} | {entry['n']} | {ge.pct(entry['accuracy_before'])} → "
                         f"{ge.pct(entry['accuracy_after'])} | {entry['fixed']}/{entry['n_wrong']} | "
                         f"{entry['broken']}/{entry['n_right']} | {entry['net_questions']:+d} questions, "
                         f"{100 * entry['net']['accuracy_change']:+.1f} pp ({ge.ci_text(entry['net']['ci'])}) | "
                         f"{report.fmt_ci(entry['prob_correct']['change'], entry['prob_correct']['ci'], decimals=4)} |")
        versus = block.get("edited_vs_control")
        if versus:
            lines += ["", "Edited minus random (counts are questions answered correctly):", "",
                      "| Questions | n | Edited | Random | Difference (CI) |", "|---|---|---|---|---|"]
            for part, label in (("wrong_fixed", "wrong before: fixed"), ("right_kept", "right before: kept"),
                                ("net", "**all**")):
                entry = versus.get(part)
                if entry is not None:
                    lines.append(f"| {label} | {entry['n']} | {entry['first']} | {entry['second']} | "
                                 f"{100 * entry['change']:+.1f} pp ({ge.ci_text(entry['ci'])}) |")
        lines += ["", f"By section ({key}):", "", "| Edit | Section | Accuracy before → after | Fixed | Broken | Net |",
                  "|---|---|---|---|---|---|"]
        for condition in CONDITIONS:
            for section, entry in (block.get(f"{condition}_by_section") or {}).items():
                lines.append(f"| {condition} | {section} | {ge.pct(entry['accuracy_before'])} → "
                             f"{ge.pct(entry['accuracy_after'])} | {entry['fixed']}/{entry['n_wrong']} | "
                             f"{entry['broken']}/{entry['n_right']} | {entry['net_questions']:+d} |")
        lines.append("")

    lines += ["### Against the unedited model and the random edit (bench.report)", "",
              "Only the test split gives a verdict that counts; gemma_extra is indicative and has no damage checks, so "
              "its damage rules are NOT MEASURED (no verdict of IMPROVEMENT without them). Headline = LÄS+MEK, metric = mean probability on the correct answer. "
              "The decision rule does not apply to edited vs random, so no verdict is shown there.", "",
              "| Set | Comparison | Verdict | Headline change (CI) | Accuracy change (pp, CI) | Wrong→right | "
              "Right→wrong | Largest letter shift | Perplexity | Loop rate |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    labels = (("edited_vs_baseline", "edited vs unedited", True), ("control_vs_baseline", "random vs unedited", True),
              ("edited_vs_control", "edited vs random", False))
    for s in ("test", "gemma_extra"):
        for key, label, with_verdict in labels:
            comparison = (numbers.get(s) or {}).get(key)
            if comparison is not None:
                lines.append(f"| {s} | {label} | {ge.comparison_cells(comparison, with_verdict=with_verdict)} |")

    test = (numbers.get("test") or {}).get("edited_vs_baseline")
    if test is not None:
        lines += ["", "By section (test split, edited vs unedited):", "",
                  "| Section | n | Accuracy change (pp, CI) | Metric change (CI) |", "|---|---|---|---|"]
        for section, group in test.get("sections", {}).items():
            lines.append(f"| {section} | {group['n']} | "
                         f"{report.fmt_ci(group['accuracy_change'], group['accuracy_ci'], scale=100)} | "
                         f"{report.fmt_ci(group['metric_change'], group['metric_ci'], decimals=4)} |")
    lines += ["", "Chosen letters: change in the share of answers choosing each letter, against the unedited model "
                  "(percentage points). An edit that only pushes the model toward one letter moves answers without "
                  "learning anything.", "", "| Set | Edit | " + " | ".join(LETTERS) + " |",
              "|---|---|" + "---|" * len(LETTERS)]
    for s in ("test", "gemma_extra"):
        for key, condition in (("edited_vs_baseline", EDITED), ("control_vs_baseline", CONTROL)):
            comparison = (numbers.get(s) or {}).get(key)
            if comparison is not None:
                shift = comparison["letter_shift"]["overall"]
                lines.append(f"| {s} | {condition} | " + " | ".join(f"{100 * shift[x]:+.1f}" for x in LETTERS) + " |")
    return lines


def evaluate_run(model: str, run: str, with_control_q4: bool = False) -> Path:
    """Steps 1 to 5 for one editor run. The full-precision model is closed before llama.cpp needs the GPU."""
    describe(model, run)
    make_control(run)
    full_precision(model, run)
    quantized(model, run, with_control=with_control_q4)
    out = collect(model, run)
    log(tag(run), "all steps finished")
    return out
