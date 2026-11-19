"""The "group edit": one small trained change that fixes many wrong answers at once.

What is trained
---------------
bench.edit fixes one question with one hand-computed change. Here we instead *train* a single
small change on a whole group of questions the model gets wrong (the "drilled wrong"
questions), and then measure what else it changed.

The change sits in the same kind of place as the single-question edit: the last matrix of the
MLP, `down_proj`, in a few neighbouring layers about two-thirds of the way up the model (by
default the 4 layers ending one above bench.edit's layer: 21-24 of Gemma 3 4B's 34, 19-22 of
Qwen3.5-9B's 32). Every other number in the model is frozen. For each of those matrices W we
learn a correction

    W  <-  W + ΔW,      ΔW = A @ B,     A: (model width x r),  B: (r x MLP width),  r = 8

Why low rank
------------
A full matrix has about 26 million entries (Gemma 3 4B); with a few hundred questions to learn
from, training all of them could simply memorise the questions. A @ B with r = 8 can only move
the layer's output inside 8 directions (the columns of A), chosen by 8 read-out patterns of the
MLP's hidden vector (the rows of B). That is about 100 thousand numbers per layer, 0.4% of the
matrix: few enough to store, compare and reason about, and it is the natural step up from
bench.edit's rank-1 change (which is r = 1 with A and B fixed by one question).

Training starts from the unedited model: B starts at zero (so ΔW = 0) and A starts as small
random numbers (columns of about unit length). The rounding of the weights is described below.

The objective
-------------
For one question the model is shown the usual prompt and we look only at the option letters:
the loss is the cross-entropy of the correct letter among the question's own letters, each
letter scored as in the HF backend (log of the summed probability of every token that is that
letter). One training step uses

    loss = mean over drilled WRONG questions in the batch
         + lam * mean over drilled RIGHT questions in the batch          (lam = 1.0)

Why the right-answer term: a change that only pushes the wrong questions towards their keys is
free to wreck everything else. Half of every batch is therefore questions the model already
answers correctly, with the same loss, so the change has to keep those answers while it fixes
the others.

Why the options are shuffled
----------------------------
If most drilled questions happened to have key "B", the cheapest thing to learn would be "say
B". To rule that out, every time a question is shown in training its options are put in a new
random order and relabelled A, B, C, ... (`shuffle_options`); the target letter follows the
correct option. No letter is favoured, so the only thing that lowers the loss is picking the
right *option*. Evaluation always uses the original order, exactly as bench.run does.

Optimiser
---------
AdamW, learning rate 1e-4, with weight decay switched off for both A and B (so it is plain
Adam). Decay would shrink A and B separately, which changes ΔW in a way that is hard to state;
instead the size of ΔW is measured and logged at every evaluation, and training stops early.
A batch is 8 questions (4 wrong, 4 right), run one question at a time with the gradients added
up, because the reading-comprehension prompts are long. Each list is walked in a fresh random
order, pass after pass, so every question is seen equally often. Everything random (the start
of A, the batch order, the option shuffles) comes from `seed`.

Only the layers from the first edited one upwards take part in the backward pass, so no
gradient checkpointing is needed (Gemma 3 4B peaks at a few GB above the model itself).

Stopping, and which change is returned
--------------------------------------
The aim is to train as far as it helps and no further. "Helps" is judged on *validation*
questions: all questions, right and wrong, of a few papers that are kept out of training (and
are not the hidden papers either). They are never trained on.

Every `eval_every` steps (and before step 1, and after the last step) the current change is
*merged into the weights* and the drilled and the validation questions are scored in their
original order with the backend's own scoring. The checkpoint with the best validation
*headline accuracy* (LÄS + MEK questions; ties go to the higher mean probability of the correct
option over all validation questions) is the one returned. The unedited model (step 0) takes
part: if no trained checkpoint beats it, the returned change is zero. Training stops when

* the best has not been beaten at `patience` (6) evaluations in a row ("no_improvement"), or
* less than 90% of the drilled right questions were kept at 3 evaluations in a row
  ("right_kept_low"), or
* `max_steps` (2000) is reached ("max_steps"), or
* `target_fix_rate` is set and that share of the drilled wrong questions is fixed (off by default).

`max_steps` = 2000 is a generous ceiling, not a target: a step takes about 1 second on Gemma
3 4B, and in the smoke runs (20 wrong + 20 right questions) the fix rate was still climbing
slowly after 400 steps, so the patience rule is what is expected to end a run.

`TrainResult.final` and `delta.info["training"]` record the returned checkpoint's step, its
validation scores, its drilled fix and kept rates, and why training stopped.

On the way, the change is also kept the first time the drilled fix rate reaches each of 80%,
85% and 90% (`TrainResult.snapshots`, keys "0.80", "0.85", "0.90"; a key is missing if that
rate was never reached). One evaluation can pass several of these at once; they are then the
same change. A snapshot that kept less than 95% of the drilled right questions is flagged
(`info["snapshot"]["kept_enough"]` is false). Snapshots are written next to the log as soon as
they exist; `save_snapshots` writes them next to a final delta.

The log (`log_path`, JSONL) has one line per evaluation: step, mean loss since the last
evaluation (total, wrong part, right part), fix rate, kept rate, the validation scores (all,
headline, per section), whether it is the best so far, and per layer the size of ΔW
(Frobenius and spectral norm, and the spectral norm as a fraction of W's own).

Numbers: bfloat16
-----------------
While training, ΔW is kept apart from W (a hook adds A B x to the layer's output) and A, B are
float32. A model that is *used* has ΔW merged into W and rounded to the weight's type, and in
bfloat16 an entry of W only changes if its share of ΔW is more than half the gap to the next
representable number; smaller shares vanish (see bench.edit). So the merged model is not quite
the model that was trained. Two things keep this honest: the stopping evaluations score the
merged, rounded weights (what `applied` and `save_merged_model` produce), not the hooked ones,
and each log line reports per layer how much of ΔW survived the rounding (`realized_fraction`,
`entries_changed`). A random control has the same Frobenius norm *before* rounding; compare
the realized sizes too (`with applied(model, delta) as change`).

`_merged_weight` is the one place where the merge arithmetic happens, for the model in memory
and for saved files alike. It adds the r rank-1 pieces one after another, each an exact IEEE
float32 operation, so the result does not depend on the device or on a matrix-product library.

torch is imported inside the functions, so `import bench.group_edit` works without it.
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Iterator

from bench.edit import (
    _backend_letter_ids,
    _checkpoint_key,
    _get_parameter,
    _log_mass,
    default_edit_layer,
    find_decoder_layers,
    find_down_proj,
    spectral_norm,
)
from bench.prompts import build_prompt, render_prompt
from bench.records import LETTERS, Question
from bench.run import score_question

if TYPE_CHECKING:
    import torch

DEFAULT_RANK = 8
# From the Gemma smoke runs (20+20 questions): 1e-4 took large first steps and never improved validation;
# 3e-5 improved it. Patience 10 evaluations (250 steps) because the full set learns more slowly.
DEFAULT_LR = 3e-5
DEFAULT_PATIENCE = 10
SNAPSHOT_RATES: tuple[float, ...] = (0.80, 0.85, 0.90)   # fix rates at which the delta is saved on the way
MIN_RIGHT_KEPT = 0.95      # below this share of drilled right questions kept, a checkpoint is flagged
RIGHT_KEPT_FLOOR = 0.90    # training stops when the kept rate is below this ...
RIGHT_KEPT_STRIKES = 3     # ... at this many evaluations in a row
HEADLINE_SECTIONS = ("LAS", "MEK")   # the validation accuracy that picks the checkpoint is over these


@dataclass
class GroupDelta:
    layers: list[int]
    param_names: list[str]          # in-memory names of the down_proj weights, one per layer
    A: list["torch.Tensor"]         # per layer (out features x rank), float32 on CPU when stored
    B: list["torch.Tensor"]         # per layer (rank x in features)
    rank: int
    kind: str                       # "trained" or "random"
    seed: int | None
    info: dict                      # {"layers": [norms per layer], "training": {...}}; plain JSON values


@dataclass
class TrainResult:
    delta: GroupDelta
    reached_target: bool            # True only if a target_fix_rate was given and reached
    steps: int                      # optimiser steps that were run (the returned delta may be from an earlier one)
    final: dict                     # the returned delta: its step, drilled rates and counts, validation scores
    log_path: Path
    # The delta the first time the fix rate reached 0.80, 0.85, 0.90: keys "0.80", "0.85", "0.90"
    # (a key is missing if that rate was never reached). Each one's info["snapshot"] has its step,
    # its rates and `kept_enough`.
    snapshots: dict[str, GroupDelta] = field(default_factory=dict)
    stop_reason: str = ""           # "no_improvement", "right_kept_low", "max_steps" or "target_fix_rate"


# --------------------------------------------------------------------------------------
# Which layers
# --------------------------------------------------------------------------------------

def default_group_layers(num_hidden_layers: int, count: int = 4) -> list[int]:
    """`count` consecutive layers ending one above bench.edit's layer (34 layers -> [21, 22, 23, 24])."""
    if not 1 <= count <= num_hidden_layers:
        raise ValueError(f"cannot pick {count} layers from a model with {num_hidden_layers}")
    last = min(default_edit_layer(num_hidden_layers) + 1, num_hidden_layers - 1)
    first = max(last - count + 1, 0)
    return list(range(first, first + count))


# --------------------------------------------------------------------------------------
# Option shuffling
# --------------------------------------------------------------------------------------

def shuffle_options(question: Question, rng: random.Random) -> Question:
    """A copy of the question with its options in a random order, relabelled A, B, C, ...

    The answer letter follows the correct option. The original question is not changed.
    """
    letters = sorted(question.options)
    shown = letters[:]          # shown[i] = the original letter of the option now at position i
    rng.shuffle(shown)
    options = {new: question.options[old] for new, old in zip(letters, shown)}
    answer = letters[shown.index(question.answer)]
    return dataclasses.replace(question, options=options, answer=answer)


# --------------------------------------------------------------------------------------
# The merge arithmetic, and sizes
# --------------------------------------------------------------------------------------

def _merged_weight(weight: "torch.Tensor", A: "torch.Tensor", B: "torch.Tensor") -> "torch.Tensor":
    """W + A @ B, computed in float32 and rounded back to the weight's own dtype.

    The product is built as a sum of rank-1 pieces in a fixed order. Each step is one exact
    float32 multiply or add per entry, so CPU and GPU give the same bits.
    """
    import torch

    A = A.to(device=weight.device, dtype=torch.float32)
    B = B.to(device=weight.device, dtype=torch.float32)
    if weight.shape != (A.shape[0], B.shape[1]) or A.shape[1] != B.shape[0]:
        raise ValueError(f"delta of shape {tuple(A.shape)} @ {tuple(B.shape)} does not fit "
                         f"a weight of shape {tuple(weight.shape)}")
    merged = weight.to(torch.float32).clone()
    for k in range(A.shape[1]):
        merged.add_(torch.outer(A[:, k], B[k]))
    return merged.to(weight.dtype)


def delta_norms(A: "torch.Tensor", B: "torch.Tensor") -> tuple[float, float]:
    """Frobenius and spectral norm of A @ B, without building the big matrix."""
    import torch

    with torch.no_grad():
        A, B = A.detach().to(torch.float32), B.detach().to(torch.float32)
        # A = Qa Ra and B^T = Qb Rb with orthonormal Q, so A @ B has the singular values of Ra Rb^T.
        core = torch.linalg.qr(A).R @ torch.linalg.qr(B.T).R.T
        singular = torch.linalg.svdvals(core)
        return float(torch.linalg.norm(singular)), float(singular.max())


def _layer_info(layer: int, A: "torch.Tensor", B: "torch.Tensor", reference_norm: float | None) -> dict:
    frobenius, spectral = delta_norms(A, B)
    return {
        "layer": layer,
        "frobenius_norm": frobenius,
        "spectral_norm": spectral,
        "reference_spectral_norm": reference_norm,      # of the unedited matrix W
        "relative_norm": spectral / reference_norm if reference_norm else None,
    }


# --------------------------------------------------------------------------------------
# Applying and removing a delta
# --------------------------------------------------------------------------------------

# model -> names of the weights that currently carry a delta. Weak keys: a dropped model is forgotten.
_APPLIED: "weakref.WeakKeyDictionary[Any, set[str]]" = weakref.WeakKeyDictionary()


@contextmanager
def applied(model, delta: GroupDelta) -> Iterator[list[dict]]:
    """Run a block with the delta merged into the weights; the exact originals are put back afterwards.

    The originals are restored from copies (not by subtracting, which would not be exact after
    rounding), also when the block raises. Yields, per layer, how much of the delta landed:

        realized_fraction  ‖W_now - W_original‖ / ‖A @ B‖ (Frobenius; 1.0 = all, 0.0 = rounded away)
        entries_changed    share of matrix entries whose stored value changed
    """
    import torch

    busy = _APPLIED.setdefault(model, set())
    overlap = busy.intersection(delta.param_names)
    if overlap:
        raise RuntimeError(f"a delta is already applied to {sorted(overlap)}; leave that block first")
    originals: dict[str, torch.Tensor] = {}
    change: list[dict] = []
    try:
        with torch.no_grad():
            for layer, name, A, B in zip(delta.layers, delta.param_names, delta.A, delta.B):
                param = _get_parameter(model, name)
                original = param.detach().clone()
                merged = _merged_weight(original, A, B)
                originals[name] = original
                busy.add(name)
                param.copy_(merged)
                change.append({"layer": layer, **_realized_change(original, merged, delta_norms(A, B)[0])})
        yield change
    finally:
        with torch.no_grad():
            for name, original in originals.items():
                _get_parameter(model, name).copy_(original)
                busy.discard(name)


def _realized_change(original: "torch.Tensor", merged: "torch.Tensor", intended_norm: float) -> dict:
    import torch

    realized = float(torch.linalg.norm(merged.to(torch.float32) - original.to(torch.float32)))
    return {
        "realized_fraction": realized / intended_norm if intended_norm > 0 else 0.0,
        "entries_changed": float((merged != original).float().mean()),
    }


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

@dataclass
class _Adapter:
    """The trainable A and B of one layer, and the hook that adds A B x to down_proj's output."""
    layer: int
    param_name: str
    module: Any
    A: "torch.Tensor"
    B: "torch.Tensor"
    reference_norm: float
    active: bool = True

    def hook(self, _module, inputs, output):
        import torch

        if not self.active:
            return None
        x = inputs[0].to(torch.float32)
        return output + ((x @ self.B.T) @ self.A.T).to(output.dtype)


def train_group_delta(backend, wrong: list[Question], right: list[Question], log_path: Path,
                      layers: list[int] | None = None, rank: int = DEFAULT_RANK, lr: float = DEFAULT_LR,
                      lam: float = 1.0, batch_size: int = 8, eval_every: int = 25, max_steps: int = 2000,
                      target_fix_rate: float | None = None, seed: int = 0,
                      validation: list[Question] | None = None, patience: int = DEFAULT_PATIENCE,
                      min_right_kept: float = MIN_RIGHT_KEPT, right_kept_floor: float = RIGHT_KEPT_FLOOR,
                      shuffle: bool = True) -> TrainResult:
    """Train one low-rank change that makes the model answer the `wrong` questions correctly
    while keeping its answers to the `right` ones, and return the checkpoint that did best on
    the `validation` questions. See the module docstring for the method and the stopping rule.

    `backend` is the full-precision Hugging Face backend (`.model`, `.tokenizer`,
    `.tokenizer_path`, `letter_probabilities`). The model is returned unedited, in eval mode,
    with no hooks attached; apply the result with `applied(backend.model, result.delta)`.

    validation       questions that are only scored, never trained on (an overlap with `wrong`
                     or `right` is an error). Without them the checkpoints are compared on the
                     drilled questions instead; see `_selection_key`.
    target_fix_rate  None (default): not a reason to stop. A number: also stop once that share
                     of the drilled wrong questions is fixed.
    shuffle          False shows every question in its original option order while training.
                     It is there for tests and as a comparison; the experiment uses the default.
    """
    import torch

    if not wrong:
        raise ValueError("there are no wrong questions to train on")
    if batch_size < 1 or (right and batch_size < 2):
        raise ValueError("batch_size must be at least 2 (one wrong and one right question)")
    validation = list(validation or [])
    _require_separate_validation(validation, wrong, right)
    model = backend.model
    if layers is None:
        layers = default_group_layers(len(find_decoder_layers(model)[1]))
    settings = {"layers": list(layers), "rank": rank, "lr": lr, "lam": lam, "batch_size": batch_size,
                "eval_every": eval_every, "max_steps": max_steps, "target_fix_rate": target_fix_rate,
                "patience": patience, "min_right_kept": min_right_kept, "right_kept_floor": right_kept_floor,
                "shuffle": shuffle, "seed": seed,
                "n_wrong": len(wrong), "n_right": len(right), "n_validation": len(validation)}

    adapters = _make_adapters(model, layers, rank, seed)
    optimizer = torch.optim.AdamW([t for a in adapters for t in (a.A, a.B)], lr=lr, weight_decay=0.0)
    letter_ids = _backend_letter_ids(backend, LETTERS)
    rng = random.Random(seed)
    wrong_cycle, right_cycle = _Cycle(wrong, rng), _Cycle(right, rng)
    n_wrong_per_batch = batch_size // 2 if right else batch_size
    n_right_per_batch = batch_size - n_wrong_per_batch

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    best: tuple[GroupDelta, dict] | None = None
    losses: list[tuple[float, float]] = []
    snapshots: dict[str, GroupDelta] = {}
    since_best = low_kept_in_a_row = step = 0
    stop_reason = None

    with open(log_path, "w", encoding="utf-8") as log, _training(model, adapters):
        while True:
            if step % eval_every == 0 or step == max_steps:
                snapshot = _snapshot(adapters, rank, seed)
                with _hooks_paused(adapters):
                    metrics = _evaluate(backend, snapshot, wrong, right, validation)
                metrics = {"step": step, **_mean_losses(losses, lam), **metrics,
                           "seconds": round(time.monotonic() - started, 1)}
                losses.clear()
                is_best = best is None or _selection_key(metrics, min_right_kept) > _selection_key(best[1], min_right_kept)
                if is_best:
                    best, since_best = (snapshot, metrics), 0
                else:
                    since_best += 1
                low_kept_in_a_row = low_kept_in_a_row + 1 if metrics["drill_right_kept"] < right_kept_floor else 0
                log.write(json.dumps({**metrics, "is_best": is_best}) + "\n")
                log.flush()
                _keep_snapshots(snapshots, snapshot, metrics, min_right_kept, log_path)
                stop_reason = _stop_reason(metrics, step, max_steps, since_best, patience,
                                           low_kept_in_a_row, target_fix_rate)
                if stop_reason:
                    break
            batch_wrong = wrong_cycle.take(n_wrong_per_batch)
            batch_right = right_cycle.take(n_right_per_batch)
            if shuffle:
                batch_wrong = [shuffle_options(q, rng) for q in batch_wrong]
                batch_right = [shuffle_options(q, rng) for q in batch_right]
            losses.append(_train_step(backend, optimizer, letter_ids, batch_wrong, batch_right, lam))
            step += 1

    delta, metrics = best
    # A copy: the returned delta may be the same checkpoint as a snapshot, and each gets its own info.
    delta = dataclasses.replace(delta, info=dict(delta.info))
    final = _drill_summary(metrics, min_right_kept)
    delta.info["layers"] = metrics["layers"]
    delta.info["training"] = {**settings, "steps_run": step, "stop_reason": stop_reason, "final": final,
                              "snapshots": {key: s.info["snapshot"] for key, s in snapshots.items()},
                              "seconds": round(time.monotonic() - started, 1)}
    return TrainResult(delta=delta, reached_target=stop_reason == "target_fix_rate", steps=step, final=final,
                       log_path=log_path, snapshots=snapshots, stop_reason=stop_reason)


def _require_separate_validation(validation: list[Question], wrong: list[Question], right: list[Question]) -> None:
    trained = {question.id for question in wrong + right}
    shared = sorted(question.id for question in validation if question.id in trained)
    if shared:
        raise ValueError(f"{len(shared)} validation questions are also trained on, for example {shared[:3]}")


def _selection_key(metrics: dict, min_right_kept: float) -> tuple:
    """What makes one checkpoint better than another (larger is better).

    With validation questions: their headline accuracy, then their mean probability of the
    correct option. Without: keeping enough of the drilled right questions, then the number of
    drilled wrong questions fixed.
    """
    validation = metrics["validation"]
    if validation is not None:
        return (validation["headline_accuracy"], validation["mean_prob_correct"])
    return (metrics["drill_right_kept"] >= min_right_kept, metrics["n_wrong_fixed"])


def _stop_reason(metrics: dict, step: int, max_steps: int, since_best: int, patience: int,
                 low_kept_in_a_row: int, target_fix_rate: float | None) -> str | None:
    """Why training ends after this evaluation, or None to go on."""
    if target_fix_rate is not None and metrics["drill_wrong_fixed"] >= target_fix_rate:
        return "target_fix_rate"
    if low_kept_in_a_row >= RIGHT_KEPT_STRIKES:
        return "right_kept_low"
    if since_best >= patience:
        return "no_improvement"
    if step >= max_steps:
        return "max_steps"
    return None


def _drill_summary(metrics: dict, min_right_kept: float) -> dict:
    """One evaluation's numbers, as stored with a delta: its step, the drilled questions, validation."""
    summary = {key: metrics[key] for key in ("step", "drill_wrong_fixed", "drill_right_kept", "n_wrong_fixed",
                                             "n_wrong", "n_right_kept", "n_right", "validation")}
    summary["kept_enough"] = metrics["drill_right_kept"] >= min_right_kept
    return summary


def snapshot_key(rate: float) -> str:
    """0.85 -> "0.85": the key of a snapshot in `TrainResult.snapshots`."""
    return f"{rate:.2f}"


def snapshot_path(path: Path, key: str) -> Path:
    """Where the snapshot `key` is saved next to `path`: runs/x/delta + "0.85" -> runs/x/delta_fix085."""
    path = Path(path)
    return path.with_name(f"{path.stem}_fix{key.replace('.', '')}")


def save_snapshots(snapshots: dict[str, GroupDelta], path: Path) -> dict[str, Path]:
    """Save every snapshot next to the final delta at `path` (see `snapshot_path`). Returns the paths by key."""
    saved = {}
    for key, snapshot in snapshots.items():
        saved[key] = snapshot_path(path, key)
        save_delta(snapshot, saved[key])
    return saved


def _keep_snapshots(snapshots: dict[str, GroupDelta], delta: GroupDelta, metrics: dict,
                    min_right_kept: float, log_path: Path) -> None:
    """Record the delta under every snapshot rate its fix rate has now reached for the first time.

    A snapshot that no longer keeps `min_right_kept` of the right questions is kept all the
    same, flagged `kept_enough: false`. Each one is also written to disk at once, next to the
    log (`<log name>_fix080.json/.safetensors`, ...), so a run that dies later still has them.
    """
    if metrics["step"] == 0:
        return   # nothing has been trained yet
    for rate in SNAPSHOT_RATES:
        key = snapshot_key(rate)
        if key in snapshots or metrics["drill_wrong_fixed"] < rate:
            continue
        info = {"layers": metrics["layers"], "snapshot": {"rate": rate, **_drill_summary(metrics, min_right_kept)}}
        snapshots[key] = dataclasses.replace(delta, info=info)
        save_delta(snapshots[key], snapshot_path(log_path, key))


def _make_adapters(model, layers: list[int], rank: int, seed: int) -> list[_Adapter]:
    """A and B for every layer: B = 0 (so the model starts unedited), A random with columns of about unit length."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    adapters = []
    for layer in layers:
        param_name, weight = find_down_proj(model, layer)
        n_out, n_in = weight.shape
        A = (torch.randn(n_out, rank, generator=generator, dtype=torch.float32) / n_out ** 0.5).to(weight.device)
        B = torch.zeros(rank, n_in, dtype=torch.float32, device=weight.device)
        adapters.append(_Adapter(layer=layer, param_name=param_name,
                                 module=model.get_submodule(param_name.removesuffix(".weight")),
                                 A=A.requires_grad_(True), B=B.requires_grad_(True),
                                 reference_norm=spectral_norm(weight)))
    return adapters


@contextmanager
def _training(model, adapters: list[_Adapter]) -> Iterator[None]:
    """Freeze the model and attach the hooks; afterwards remove them and put everything back."""
    params = list(model.parameters())
    was_trainable = [p.requires_grad for p in params]
    handles = []
    try:
        for p in params:
            p.requires_grad_(False)
        model.eval()   # no dropout: the training forward pass is the scoring forward pass plus the hooks
        handles = [adapter.module.register_forward_hook(adapter.hook) for adapter in adapters]
        yield
    finally:
        for handle in handles:
            handle.remove()
        for p, flag in zip(params, was_trainable):
            p.requires_grad_(flag)
        model.eval()


@contextmanager
def _hooks_paused(adapters: list[_Adapter]) -> Iterator[None]:
    for adapter in adapters:
        adapter.active = False
    try:
        yield
    finally:
        for adapter in adapters:
            adapter.active = True


class _Cycle:
    """Hands out the items in a random order, pass after pass, reshuffling for every pass."""

    def __init__(self, items: list, rng: random.Random):
        self._items, self._rng, self._queue = list(items), rng, []

    def take(self, count: int) -> list:
        taken = []
        while len(taken) < count and self._items:
            if not self._queue:
                self._queue = self._items[:]
                self._rng.shuffle(self._queue)
            taken.append(self._queue.pop())
        return taken


def _train_step(backend, optimizer, letter_ids: dict[str, list[int]], batch_wrong: list[Question],
                batch_right: list[Question], lam: float) -> tuple[float, float]:
    """One optimiser step on one batch, one question per forward pass. Returns the two mean losses."""
    import torch

    optimizer.zero_grad(set_to_none=True)
    means = []
    for questions, weight in ((batch_wrong, 1.0), (batch_right, lam)):
        total = 0.0
        for question in questions:
            with torch.enable_grad():
                loss = letter_cross_entropy(backend, question, letter_ids)
                (loss * (weight / len(questions))).backward()
            total += loss.item()
        means.append(total / len(questions) if questions else 0.0)
    if not all(torch.isfinite(torch.tensor(means))):
        raise RuntimeError(f"the training loss is not a finite number: wrong {means[0]}, right {means[1]}")
    optimizer.step()
    return means[0], means[1]


def letter_cross_entropy(backend, question: Question, letter_ids: dict[str, list[int]] | None = None,
                         ) -> "torch.Tensor":
    """Cross-entropy of the correct letter among the question's own option letters, on the model as it is now.

    Each letter's score is the log of the summed probability of its tokens (the HF backend's
    rule); the softmax is taken over the question's letters only.
    """
    import torch

    model = backend.model
    letters = sorted(question.options)
    if letter_ids is None:
        letter_ids = _backend_letter_ids(backend, tuple(letters))
    prompt = render_prompt(build_prompt(question), str(backend.tokenizer_path))
    # add_special_tokens=False: the chat template already wrote BOS, as in the backend's scoring.
    input_ids = backend.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    logits = model(input_ids=input_ids.to(model.device), logits_to_keep=1).logits[0, -1].to(torch.float32)
    scores = torch.stack([_log_mass(logits, letter_ids[letter]) for letter in letters])
    return -torch.log_softmax(scores, dim=0)[letters.index(question.answer)]


def _evaluate(backend, delta: GroupDelta, wrong: list[Question], right: list[Question],
              validation: list[Question]) -> dict:
    """Score the drilled and the validation questions, in their original order, with the delta merged into the weights."""
    tokenizer_path = str(backend.tokenizer_path)
    with applied(backend.model, delta) as change:
        n_fixed = sum(score_question(backend, q, tokenizer_path).is_correct for q in wrong)
        n_kept = sum(score_question(backend, q, tokenizer_path).is_correct for q in right)
        validation_answers = [score_question(backend, q, tokenizer_path) for q in validation]
    return {
        "drill_wrong_fixed": n_fixed / len(wrong),
        "drill_right_kept": n_kept / len(right) if right else 1.0,
        "n_wrong_fixed": n_fixed, "n_wrong": len(wrong), "n_right_kept": n_kept, "n_right": len(right),
        "validation": _validation_scores(validation_answers),
        "layers": [{**info, **{k: v for k, v in landed.items() if k != "layer"}}
                   for info, landed in zip(delta.info["layers"], change)],
    }


def _validation_scores(answers: list) -> dict | None:
    """Accuracy on the validation questions: all of them, per section, and the headline (LAS + MEK).

    If there are no LAS or MEK questions among them, the headline is the accuracy on all.
    """
    if not answers:
        return None
    headline = [a for a in answers if a.section in HEADLINE_SECTIONS] or answers
    sections = sorted({a.section for a in answers})
    return {
        "n": len(answers),
        "accuracy": sum(a.is_correct for a in answers) / len(answers),
        "n_headline": len(headline),
        "headline_accuracy": sum(a.is_correct for a in headline) / len(headline),
        "mean_prob_correct": sum(a.prob_correct for a in answers) / len(answers),
        "by_section": {s: sum(a.is_correct for a in answers if a.section == s) / sum(a.section == s for a in answers)
                       for s in sections},
    }


def _mean_losses(losses: list[tuple[float, float]], lam: float) -> dict:
    """Mean training loss over the steps since the last evaluation (None before the first step)."""
    if not losses:
        return {"loss": None, "loss_wrong": None, "loss_right": None}
    loss_wrong = sum(pair[0] for pair in losses) / len(losses)
    loss_right = sum(pair[1] for pair in losses) / len(losses)
    return {"loss": loss_wrong + lam * loss_right, "loss_wrong": loss_wrong, "loss_right": loss_right}


def _snapshot(adapters: list[_Adapter], rank: int, seed: int) -> GroupDelta:
    """The adapters' current A and B as a stored delta (copies on the CPU)."""
    A = [adapter.A.detach().cpu().clone() for adapter in adapters]
    B = [adapter.B.detach().cpu().clone() for adapter in adapters]
    info = {"layers": [_layer_info(adapter.layer, a, b, adapter.reference_norm)
                       for adapter, a, b in zip(adapters, A, B)]}
    return GroupDelta(layers=[adapter.layer for adapter in adapters],
                      param_names=[adapter.param_name for adapter in adapters],
                      A=A, B=B, rank=rank, kind="trained", seed=seed, info=info)


# --------------------------------------------------------------------------------------
# The random control
# --------------------------------------------------------------------------------------

def random_delta_like(delta: GroupDelta, seed: int) -> GroupDelta:
    """The control: same layers, same rank, and per layer the same Frobenius norm of A @ B,
    but A and B are random Gaussian matrices (repeatable from `seed`).

    The norm is matched before the weights are rounded; see `applied` for what lands.
    """
    import torch

    generator = torch.Generator().manual_seed(seed)
    A_list, B_list, layer_infos = [], [], []
    references = {entry.get("layer"): entry.get("reference_spectral_norm") for entry in delta.info.get("layers", [])}
    for layer, trained_A, trained_B in zip(delta.layers, delta.A, delta.B):
        A = torch.randn(trained_A.shape, generator=generator, dtype=torch.float32)
        B = torch.randn(trained_B.shape, generator=generator, dtype=torch.float32)
        target = delta_norms(trained_A, trained_B)[0]
        A = A * (target / delta_norms(A, B)[0])
        A_list.append(A)
        B_list.append(B)
        layer_infos.append(_layer_info(layer, A, B, references.get(layer)))
    return GroupDelta(layers=list(delta.layers), param_names=list(delta.param_names), A=A_list, B=B_list,
                      rank=delta.rank, kind="random", seed=seed,
                      info={"layers": layer_infos, "control_for": {"kind": delta.kind, "seed": delta.seed}})


# --------------------------------------------------------------------------------------
# Saving and loading a delta
# --------------------------------------------------------------------------------------

def save_delta(delta: GroupDelta, path: Path) -> None:
    """Write the delta as `<path>.json` (everything but the matrices) and `<path>.safetensors` (A.<i>, B.<i>)."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(_metadata(delta), indent=2) + "\n", encoding="utf-8")
    tensors = {}
    for index, (A, B) in enumerate(zip(delta.A, delta.B)):
        tensors[f"A.{index}"] = A.detach().cpu().float().contiguous()
        tensors[f"B.{index}"] = B.detach().cpu().float().contiguous()
    save_file(tensors, str(path.with_suffix(".safetensors")))


def load_delta(path: Path) -> GroupDelta:
    """Read a delta written by `save_delta` (same `path`, with or without a suffix)."""
    from safetensors.torch import load_file

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    tensors = load_file(str(path.with_suffix(".safetensors")))
    count = len(metadata["layers"])
    return GroupDelta(A=[tensors[f"A.{i}"] for i in range(count)], B=[tensors[f"B.{i}"] for i in range(count)],
                      **metadata)


def _metadata(delta: GroupDelta) -> dict[str, Any]:
    """The delta's fields without the matrices."""
    return {name: getattr(delta, name) for name in delta.__dataclass_fields__ if name not in ("A", "B")}


# --------------------------------------------------------------------------------------
# Writing a merged model folder
# --------------------------------------------------------------------------------------

def save_merged_model(src_dir: Path, delta: GroupDelta, out_dir: Path) -> Path:
    """Write a loadable Hugging Face model folder that is `src_dir` with the delta merged in.

    Like bench.edit.save_edited_model: small files are copied, weight shards without an edited
    matrix are symlinks to the originals, and only the shards that hold an edited matrix are
    rewritten (under the same file names, so the shard index stays valid). The matrices are
    changed by `_merged_weight`, the function `applied` uses, in the dtype stored in the file;
    when that is the dtype the model is run in, the file holds exactly the numbers of the model
    in memory under `applied` (tested bit for bit).

    Names inside the files can differ from the names in the loaded model (Gemma 3); that is
    resolved as in bench.edit. The folder is built next to `out_dir` and renamed at the end, so
    an interrupted call never leaves a half-written `out_dir`. Returns `out_dir`.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")
    shards = sorted(src_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {src_dir}")
    # shard -> [(name in the file, index of the layer in the delta)]
    todo: dict[Path, list[tuple[str, int]]] = {}
    for index, param_name in enumerate(delta.param_names):
        key, shard = _checkpoint_key(shards, SimpleNamespace(param_name=param_name))
        todo.setdefault(shard, []).append((key, index))

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
        elif entry not in todo:
            target.symlink_to(entry.resolve())

    rewritten = []
    for shard, entries in todo.items():
        with safe_open(str(shard), framework="pt") as f:
            file_metadata = f.metadata()
        tensors = load_file(str(shard))
        for key, index in entries:
            original = tensors[key]
            tensors[key] = _merged_weight(original, delta.A[index], delta.B[index])
            landed = _realized_change(original, tensors[key], delta_norms(delta.A[index], delta.B[index])[0])
            rewritten.append({"layer": delta.layers[index], "param_name": delta.param_names[index],
                              "checkpoint_key": key, "file": shard.name, **landed})
        save_file(tensors, str(partial / shard.name), metadata=file_metadata)

    provenance = {**_metadata(delta), "source_model": str(src_dir), "rewritten": rewritten}
    (partial / "selma_group_edit.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    partial.rename(out_dir)
    return out_dir
