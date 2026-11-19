"""How we change a model's weights: one rank-1 edit to one MLP output matrix.

The idea
--------
Every transformer layer has an MLP whose last step is a matrix, `down_proj`, that maps the
MLP's wide hidden vector `x` (MLP hidden size) to a vector that is added to the model's
running state (model width):  out = W x.

To fix one question the model answers wrongly we ask: at the position where the model is
about to write its answer letter, in which direction should this layer's MLP output move so
that the correct letter gains on the letter the model wrongly chose? That direction is the
gradient `g` of the margin

    m = log p(correct letter) - log p(wrongly chosen letter)

with respect to the MLP output at that one position. We then change W so that, for exactly
the hidden vector `x` seen at that position, the output moves along `g`:

    W  <-  W + alpha * u ⊗ v,      u = g / ‖g‖,   v = x / ‖x‖

For this prompt the MLP output at the last position moves by alpha * ‖x‖ * u, so the margin
rises (to first order) by alpha * ‖x‖ * ‖g‖. For any other input x' the output moves by
alpha * (v · x') * u: inputs that look like `x` are pushed the same way, inputs unrelated to
`x` are left alone. Whether "inputs that look like x" means "this question", "Swedish
questions like it" or simply "any prompt that ends with the same answer template" is exactly
what the experiment measures; this module only makes the edit.

It is one forward and one backward pass. There is no optimizer and no training loop.

Which layer
-----------
A fixed layer about two-thirds of the way up the text model: `default_edit_layer(n)` =
round(2n/3), counted from 0. That is layer 21 of 32 for Qwen3.5-9B and layer 23 of 34 for
Gemma 3 4B (recorded as `edit_layer` in config/models.toml). `find_down_proj` locates the
matrix in any model whose decoder layers each have `.mlp.down_proj`.

How large (alpha)
-----------------
alpha is the smallest step on a fixed ladder that makes the model answer the target question
correctly, checked by really applying the edit and scoring the question the way bench.run
does. The ladder is in fractions of the *spectral norm* of W (its largest singular value,
i.e. the most W can stretch any input):

    LADDER = 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0   (alpha = fraction * ‖W‖₂)

Why the spectral norm and not the Frobenius norm: u ⊗ v has spectral norm 1, so the fraction
compares like with like ("the edit's strongest effect is this fraction of the matrix's
strongest effect"), and it gives a guarantee that does not depend on the matrix's size. The
edit moves the MLP output at the target position by alpha * ‖x‖, while the unedited output
‖W x‖ is at most ‖W‖₂ * ‖x‖, so

    ‖ΔW x‖ / ‖W x‖  =  fraction * (‖W‖₂ ‖x‖ / ‖W x‖)  >=  fraction.

The Frobenius norm grows with the square root of the matrix's rank (about 64x the typical
singular value for a 4096-wide matrix), so the same fractions of it would start at a push
that already swamps the MLP output.

Scale check behind the ladder (done once, on paper and on the tiny test model, before any
real model was edited; the ladder is fixed from here on): by the inequality above the bottom
step changes this MLP's output at the answer position by at least 1%, the top step by at
least 100%. Real hidden vectors are usually stretched several times less than the worst
case, so the real pushes are a few times larger: the ladder runs from a nudge to an edit
that overrides the layer's MLP at that position. That covers the useful range, so the
suggested ladder was kept unchanged. The actual ratio for every target is stored in
`EditSearch.baseline["push_ratio_per_fraction"]`, so the assumption can be checked afterwards.

Numbers: bfloat16
-----------------
The real models hold their weights in bfloat16 (8 significant bits). The edit is computed in
float32, added, and rounded back to the weight's own type (`_edited_weight`, used both for
the model in memory and for the saved file, so the two agree bit for bit). Rounding has two
consequences:

* Undoing an edit by subtracting it would not give back the original weights, so
  `apply_edit` keeps a copy of the original matrix and `remove_edit` puts that copy back.
* A small edit can partly or wholly round away: an entry only changes if its share of the
  edit is more than half the spacing between neighbouring bfloat16 numbers (about 0.2-0.4%
  of the entry's own size). `applied_change` reports how much of the intended edit actually
  landed (`realized_fraction`, `entries_changed`); the ladder search records it for every
  step, and `apply_edit` warns when nothing changed at all. A random control with the same
  alpha spreads its change evenly over all entries, so it can lose more to rounding than a
  targeted edit whose `x` and `g` are concentrated in few entries; compare the realized
  sizes, not just alpha.

torch is imported inside the functions, so `import bench.edit` works without it.
"""

from __future__ import annotations

import json
import math
import shutil
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from bench.prompts import build_prompt, render_prompt
from bench.records import LETTERS, Question
from bench.run import score_answer

if TYPE_CHECKING:
    import torch

# Fractions of the spectral norm of the edited matrix. Fixed; see the module docstring.
LADDER: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
REFERENCE_NORM = "spectral"

# Probabilities below this are treated as this when taking logs for the reported margin.
_PROB_FLOOR = 1e-30


@dataclass
class Edit:
    layer: int
    param_name: str          # full parameter name of the down_proj weight in model.named_parameters()
    u: "torch.Tensor"        # unit vector, length = out features (model width), float32 on CPU when stored
    v: "torch.Tensor"        # unit vector, length = in features (MLP hidden size)
    alpha: float             # absolute scale (not the fraction)
    alpha_fraction: float    # alpha / reference norm
    kind: str                # "targeted" or "random"
    question_id: str | None  # the question the edit was made for (kept on its random control too)
    seed: int | None         # random edits only


@dataclass
class EditSearch:
    question_id: str
    fixed: bool
    edit: Edit | None        # the smallest edit that fixed the question, or None
    tried: list[dict]        # per ladder step: alpha_fraction, chosen_letter, margin, prob_correct, ...
    status: str = ""         # "fixed", "could_not_fix", "already_correct" or "no_gradient"
    baseline: dict = field(default_factory=dict)   # the unedited model on this question, and the scale of x, g, W


# --------------------------------------------------------------------------------------
# Finding the matrix
# --------------------------------------------------------------------------------------

def default_edit_layer(num_hidden_layers: int) -> int:
    """The fixed edit layer: about two-thirds of the way up, counted from 0 (32 -> 21, 34 -> 23)."""
    return round(2 * num_hidden_layers / 3)


def find_decoder_layers(model) -> tuple[str, "torch.nn.ModuleList"]:
    """The text model's list of decoder layers, and its dotted path inside `model`.

    A decoder layer list is a ModuleList in which every element has `.mlp.down_proj` as a
    Linear. Vision towers do not match (their MLPs have fc1/fc2). If a model has more than
    one such list (Qwen3.5 has a one-layer extra head, `mtp.layers`), the one whose length
    equals the text config's num_hidden_layers is taken. Anything else is an error.
    """
    import torch

    candidates = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0
        and all(_has_down_proj(layer) for layer in module)
    ]
    if len(candidates) > 1:
        expected = _text_num_hidden_layers(model)
        candidates = [(name, module) for name, module in candidates if len(module) == expected]
    if len(candidates) != 1:
        found = ", ".join(name for name, _ in candidates) or "none"
        raise ValueError(
            "could not identify the decoder layer list: expected exactly one ModuleList whose "
            f"elements all have .mlp.down_proj, found {found}"
        )
    return candidates[0]


def find_down_proj(model, layer: int) -> tuple[str, "torch.nn.Parameter"]:
    """Parameter name and parameter of the MLP output matrix of decoder layer `layer` (0-based)."""
    layers_path, layers = find_decoder_layers(model)
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer {layer} is out of range: the model has layers 0..{len(layers) - 1}")
    param_name = f"{layers_path}.{layer}.mlp.down_proj.weight"
    return param_name, _get_parameter(model, param_name)


def _has_down_proj(layer) -> bool:
    import torch

    mlp = getattr(layer, "mlp", None)
    return isinstance(getattr(mlp, "down_proj", None), torch.nn.Linear)


def _text_num_hidden_layers(model) -> int | None:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    return getattr(text_config, "num_hidden_layers", None)


def _get_parameter(model, param_name: str) -> "torch.nn.Parameter":
    try:
        return model.get_parameter(param_name)
    except AttributeError as error:
        raise ValueError(f"the model has no parameter named {param_name!r}") from error


# --------------------------------------------------------------------------------------
# Applying and removing an edit
# --------------------------------------------------------------------------------------

# model -> {param_name: untouched copy of the weight}, for every edit currently applied.
# Weak keys, so a model that is dropped takes its copies with it.
_ORIGINALS: "weakref.WeakKeyDictionary[Any, dict[str, torch.Tensor]]" = weakref.WeakKeyDictionary()


def _edited_weight(weight: "torch.Tensor", edit: Edit) -> "torch.Tensor":
    """W + alpha * u ⊗ v, computed in float32 and rounded back to the weight's own dtype.

    The single place where the edit arithmetic happens: `apply_edit` (model in memory) and
    `save_edited_model` (file on disk) both call it, so they produce the same numbers.
    """
    import torch

    u = edit.u.to(device=weight.device, dtype=torch.float32)
    v = edit.v.to(device=weight.device, dtype=torch.float32)
    if weight.shape != (u.numel(), v.numel()):
        raise ValueError(f"edit has shape ({u.numel()}, {v.numel()}) but the weight "
                         f"{edit.param_name} has shape {tuple(weight.shape)}")
    delta = torch.outer(u, v).mul_(edit.alpha)
    return (weight.to(torch.float32) + delta).to(weight.dtype)


def apply_edit(model, edit: Edit) -> None:
    """W += alpha * u ⊗ v on the model in memory, in the weight's dtype and on its device.

    Keeps a copy of the original matrix so `remove_edit` can restore it exactly. Only one
    edit per matrix can be applied at a time. Warns if rounding swallowed the whole edit.
    """
    import torch

    param = _get_parameter(model, edit.param_name)
    originals = _ORIGINALS.setdefault(model, {})
    if edit.param_name in originals:
        raise RuntimeError(f"an edit is already applied to {edit.param_name}; remove it first")
    with torch.no_grad():
        original = param.detach().clone()
        param.copy_(_edited_weight(original, edit))
        originals[edit.param_name] = original
        if torch.equal(param, original):
            warnings.warn(
                f"edit with alpha={edit.alpha:g} left {edit.param_name} unchanged: "
                f"it rounded away in {param.dtype}", RuntimeWarning, stacklevel=2,
            )


def remove_edit(model, edit: Edit) -> None:
    """Put back the original matrix (the saved copy, not a subtraction, so it is exact)."""
    import torch

    originals = _ORIGINALS.get(model, {})
    if edit.param_name not in originals:
        raise RuntimeError(f"no edit is applied to {edit.param_name}")
    with torch.no_grad():
        _get_parameter(model, edit.param_name).copy_(originals.pop(edit.param_name))


@contextmanager
def edited(model, edit: Edit) -> Iterator[None]:
    """Run a block with the edit applied; the original weights are always restored afterwards."""
    apply_edit(model, edit)
    try:
        yield
    finally:
        remove_edit(model, edit)


def applied_change(model, edit: Edit) -> dict:
    """How much of the intended edit actually landed in the weights. Call while the edit is applied.

    realized_fraction  ‖W_now - W_original‖ / alpha (Frobenius; 1.0 = all of it, 0.0 = rounded away)
    realized_cosine    direction agreement between the actual and the intended change (1.0 = same)
    entries_changed    share of matrix entries whose stored value changed
    weight_changed     False if the edit rounded away completely
    """
    import torch

    originals = _ORIGINALS.get(model, {})
    if edit.param_name not in originals:
        raise RuntimeError(f"no edit is applied to {edit.param_name}")
    with torch.no_grad():
        original = originals[edit.param_name]
        current = _get_parameter(model, edit.param_name).detach()
        realized = current.to(torch.float32) - original.to(torch.float32)
        realized_norm = float(torch.linalg.norm(realized))
        # <realized, u ⊗ v> = u · (realized v), without building the outer product again.
        u = edit.u.to(device=realized.device, dtype=torch.float32)
        v = edit.v.to(device=realized.device, dtype=torch.float32)
        along_intended = float(u @ (realized @ v))
        entries_changed = float((current != original).float().mean())
    return {
        "weight_changed": realized_norm > 0,
        "realized_fraction": realized_norm / abs(edit.alpha) if edit.alpha else 0.0,
        "realized_cosine": along_intended / realized_norm * math.copysign(1.0, edit.alpha) if realized_norm > 0 else 0.0,
        "entries_changed": entries_changed,
    }


# --------------------------------------------------------------------------------------
# Computing the targeted edit
# --------------------------------------------------------------------------------------

def compute_targeted_edit(backend, question: Question, layer: int,
                          ladder: list[float] | None = None,
                          reference_norm: float | None = None) -> EditSearch:
    """Find the smallest ladder step of the rank-1 edit that makes the model answer `question` correctly.

    `backend` is the full-precision Hugging Face backend: it must have `.model`,
    `.tokenizer`, `.tokenizer_path` and `letter_probabilities`. The model is left unedited
    when this returns; apply the returned edit with `edited(...)` or `apply_edit(...)`.

    Steps:
    1. Score the question on the unedited model (same prompt and scoring as bench.run).
       If it is already correct there is nothing to fix: status "already_correct".
    2. One forward and backward pass gives x (input of down_proj at the last prompt
       position) and g (gradient of the margin with respect to its output there).
    3. For each fraction on the ladder, smallest first: apply alpha = fraction * ‖W‖₂,
       score the question again, remove the edit. Stop at the first step that fixes it.

    `reference_norm` (the spectral norm of the unedited matrix) can be passed in to avoid
    recomputing it for every question of the same model and layer; see `spectral_norm`.
    """
    model = backend.model
    fractions = list(LADDER if ladder is None else ladder)
    prompt = render_prompt(build_prompt(question), str(backend.tokenizer_path))
    param_name, weight = find_down_proj(model, layer)

    before = _score(backend, question, prompt)
    wrong_letter = before.chosen_letter
    baseline: dict[str, Any] = {
        "chosen_letter": wrong_letter,
        "prob_correct": before.prob_correct,
        "margin": _margin_from_probs(before.letter_probs, question.answer, wrong_letter),
        "layer": layer,
        "param_name": param_name,
    }
    if before.is_correct:
        return EditSearch(question.id, fixed=False, edit=None, tried=[], status="already_correct", baseline=baseline)

    x, g, mlp_out_norm = _input_and_margin_gradient(backend, prompt, param_name, question.answer, wrong_letter)
    x_norm, g_norm = float(x.norm()), float(g.norm())
    sigma = spectral_norm(weight) if reference_norm is None else float(reference_norm)
    baseline.update({
        "reference_norm": sigma,
        "reference_norm_kind": REFERENCE_NORM,
        "x_norm": x_norm,
        "g_norm": g_norm,
        "mlp_out_norm": mlp_out_norm,
        # ‖ΔW x‖ / ‖W x‖ at the answer position for alpha_fraction = 1; multiply by the fraction.
        "push_ratio_per_fraction": sigma * x_norm / mlp_out_norm if mlp_out_norm > 0 else None,
    })
    if not (math.isfinite(x_norm) and math.isfinite(g_norm)) or x_norm == 0 or g_norm == 0:
        return EditSearch(question.id, fixed=False, edit=None, tried=[], status="no_gradient", baseline=baseline)

    u, v = (g / g_norm).cpu(), (x / x_norm).cpu()
    tried: list[dict] = []
    for fraction in fractions:
        edit = Edit(layer=layer, param_name=param_name, u=u, v=v, alpha=fraction * sigma,
                    alpha_fraction=fraction, kind="targeted", question_id=question.id, seed=None)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # a rounded-away step is recorded below instead
            with edited(model, edit):
                change = applied_change(model, edit)
                after = _score(backend, question, prompt)
        tried.append({
            "alpha_fraction": fraction,
            "alpha": edit.alpha,
            "chosen_letter": after.chosen_letter,
            "is_correct": after.is_correct,
            "margin": _margin_from_probs(after.letter_probs, question.answer, wrong_letter),
            "prob_correct": after.prob_correct,
            **change,
        })
        if after.is_correct:
            return EditSearch(question.id, fixed=True, edit=edit, tried=tried, status="fixed", baseline=baseline)
    return EditSearch(question.id, fixed=False, edit=None, tried=tried, status="could_not_fix", baseline=baseline)


def spectral_norm(weight: "torch.Tensor") -> float:
    """Largest singular value of the matrix, computed in float32. This is the ladder's unit."""
    import torch

    with torch.no_grad():
        return float(torch.linalg.matrix_norm(weight.detach().to(torch.float32), ord=2))


def random_edit_like(edit: Edit, seed: int) -> Edit:
    """The control: same layer, same alpha, but u and v are random unit vectors (repeatable from `seed`)."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    u = torch.randn(edit.u.numel(), generator=generator, dtype=torch.float32)
    v = torch.randn(edit.v.numel(), generator=generator, dtype=torch.float32)
    return Edit(layer=edit.layer, param_name=edit.param_name, u=u / u.norm(), v=v / v.norm(),
                alpha=edit.alpha, alpha_fraction=edit.alpha_fraction, kind="random",
                question_id=edit.question_id, seed=seed)


def letter_token_ids(tokenizer, letters: tuple[str, ...] = LETTERS) -> dict[str, list[int]]:
    """Every vocabulary id whose decoded text, stripped of whitespace, is exactly one letter.

    The same rule the HF backend scores with ("A" and " A" both count as A).
    """
    found: dict[str, list[int]] = {letter: [] for letter in letters}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id]).strip()
        if text in found:
            found[text].append(token_id)
    return found


def _score(backend, question: Question, prompt: str):
    """The question's Answer on the model as it is right now, scored exactly as bench.run does."""
    raw_probs = backend.letter_probabilities(prompt, sorted(question.options), question)
    return score_answer(question, raw_probs)


def _margin_from_probs(letter_probs: dict[str, float], correct: str, wrong: str) -> float:
    """log p(correct) - log p(wrong); positive means the correct letter is ahead of that wrong one."""
    return (math.log(max(letter_probs.get(correct, 0.0), _PROB_FLOOR))
            - math.log(max(letter_probs.get(wrong, 0.0), _PROB_FLOOR)))


def _backend_letter_ids(backend, letters: tuple[str, ...]) -> dict[str, list[int]]:
    """Token ids per letter: the backend's own table when it has one, else the same rule recomputed."""
    table = getattr(backend, "_letter_ids", None)
    if table is not None and all(letter in table for letter in letters):
        return {letter: [int(i) for i in table[letter]] for letter in letters}
    return letter_token_ids(backend.tokenizer, letters)


def _input_and_margin_gradient(backend, prompt: str, param_name: str, correct: str, wrong: str,
                               ) -> tuple["torch.Tensor", "torch.Tensor", float]:
    """One forward and one backward pass: x, g and ‖W x‖ at the last prompt position.

    x  the vector entering down_proj at the last position (float32)
    g  gradient of the margin with respect to down_proj's output at the last position (float32)

    A hook on down_proj reads x and swaps the output's last position for a fresh tensor
    that tracks gradients. Everything below the hook runs without gradients (the weights
    are frozen for the duration), so the backward pass only covers the layers above it and
    yields the gradient for that single position.
    """
    import torch

    model = backend.model
    module = model.get_submodule(param_name.removesuffix(".weight"))
    letter_ids = _backend_letter_ids(backend, (correct, wrong))
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, output):
        captured["x"] = inputs[0][0, -1].detach().to(torch.float32)
        captured["leaf"] = output[0, -1].detach().clone().requires_grad_(True)
        replaced = output.detach().clone()
        replaced[0, -1] = captured["leaf"]
        return replaced

    # add_special_tokens=False: the chat template already wrote BOS, as in the backend's scoring.
    input_ids = backend.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
    params = list(model.parameters())
    was_trainable = [p.requires_grad for p in params]
    handle = module.register_forward_hook(hook)
    try:
        for p in params:
            p.requires_grad_(False)
        with torch.enable_grad():
            logits = model(input_ids=input_ids.to(model.device), logits_to_keep=1).logits[0, -1].to(torch.float32)
            margin = _log_mass(logits, letter_ids[correct]) - _log_mass(logits, letter_ids[wrong])
            (gradient,) = torch.autograd.grad(margin, captured["leaf"])
    finally:
        handle.remove()
        for p, flag in zip(params, was_trainable):
            p.requires_grad_(flag)
    mlp_out_norm = float(captured["leaf"].detach().to(torch.float32).norm())
    return captured["x"], gradient.detach().to(torch.float32), mlp_out_norm


def _log_mass(logits: "torch.Tensor", token_ids: list[int]) -> "torch.Tensor":
    """Log of the summed probability of these tokens, up to the constant shared by all tokens.

    The constant (the softmax normaliser) cancels in the margin. With one token per letter
    this is just that token's logit.
    """
    import torch

    in_range = [i for i in token_ids if i < logits.numel()]   # ids beyond the output size have no probability
    if not in_range:
        raise ValueError("a letter has no token in the model's vocabulary")
    return torch.logsumexp(logits[in_range], dim=0)


# --------------------------------------------------------------------------------------
# Saving and loading an edit
# --------------------------------------------------------------------------------------

def save_edit(edit: Edit, path: Path) -> None:
    """Write the edit as `<path>.json` (everything but the vectors) and `<path>.safetensors` (u and v)."""
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in asdict_shallow(edit).items() if key not in ("u", "v")}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save_file({"u": edit.u.detach().cpu().float().contiguous(), "v": edit.v.detach().cpu().float().contiguous()},
              str(path.with_suffix(".safetensors")))


def load_edit(path: Path) -> Edit:
    """Read an edit written by `save_edit` (same `path`, with or without a suffix)."""
    from safetensors.torch import load_file

    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    vectors = load_file(str(path.with_suffix(".safetensors")))
    return Edit(u=vectors["u"], v=vectors["v"], **metadata)


def asdict_shallow(edit: Edit) -> dict[str, Any]:
    """The edit's fields as a dict, without copying the tensors (dataclasses.asdict would deep-copy them)."""
    return {name: getattr(edit, name) for name in edit.__dataclass_fields__}


# --------------------------------------------------------------------------------------
# Writing an edited model folder
# --------------------------------------------------------------------------------------

def save_edited_model(src_dir: Path, edit: Edit, out_dir: Path) -> Path:
    """Write a loadable Hugging Face model folder that is `src_dir` with the edit applied.

    * Small files (config, tokenizer, chat template, the shard index, ...) are copied.
    * Every weight shard that does not hold the edited matrix is a symlink to the original.
    * The one shard that holds it is rewritten with only that matrix changed. The shard
      keeps its file name, so model.safetensors.index.json stays valid as it is.

    The matrix is changed by `_edited_weight`, the same function `apply_edit` uses, in the
    dtype stored in the file. The stored dtype of both models is bfloat16, the dtype they
    are run in, so the file holds the same numbers as the edited model in memory (tested
    bit for bit on CPU; float32 multiply and add and the rounding to bfloat16 are exact
    IEEE operations, so the GPU result should be identical, but that is not tested here).

    The name of a weight inside the files can differ from its name in the loaded model
    (Gemma 3 stores `language_model.model.layers...`, transformers loads it as
    `model.language_model.layers...`); `_checkpoint_key` handles that.

    The folder is built next to `out_dir` and renamed at the end, so an interrupted call
    never leaves a half-written `out_dir`. Returns `out_dir`.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    src_dir, out_dir = Path(src_dir), Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists")
    shards = sorted(src_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors files in {src_dir}")
    key, edited_shard = _checkpoint_key(shards, edit)

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
        elif entry != edited_shard:
            target.symlink_to(entry.resolve())

    with safe_open(str(edited_shard), framework="pt") as f:
        file_metadata = f.metadata()
    tensors = load_file(str(edited_shard))
    original = tensors[key]
    tensors[key] = _edited_weight(original, edit)
    weight_changed = not bool((tensors[key] != original).sum() == 0)
    save_file(tensors, str(partial / edited_shard.name), metadata=file_metadata)

    provenance = {key_: value for key_, value in asdict_shallow(edit).items() if key_ not in ("u", "v")}
    provenance.update({"source_model": str(src_dir), "checkpoint_key": key,
                       "rewritten_file": edited_shard.name, "weight_changed": weight_changed})
    (partial / "selma_edit.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    if not weight_changed:
        warnings.warn(f"edit with alpha={edit.alpha:g} left {key} unchanged in the saved model: "
                      f"it rounded away in {original.dtype}", RuntimeWarning, stacklevel=2)

    partial.rename(out_dir)
    return out_dir


def _checkpoint_key(shards: list[Path], edit: Edit) -> tuple[str, Path]:
    """The name under which the edited matrix is stored on disk, and the shard that holds it.

    Usually the parameter name itself. Otherwise the stored name must end in the same
    `layers.<n>.mlp.down_proj.weight` and have the same leading parts in a different order
    (the Gemma 3 case), or be the only stored name with that ending.
    """
    from safetensors import safe_open

    tail = "layers." + edit.param_name.split("layers.")[-1]
    head_parts = sorted(part for part in edit.param_name[: -len(tail)].split(".") if part)
    where: dict[str, Path] = {}
    for shard in shards:
        with safe_open(str(shard), framework="pt") as f:
            for key in f.keys():
                if key == tail or key.endswith("." + tail):
                    where[key] = shard
    if edit.param_name in where:
        return edit.param_name, where[edit.param_name]
    same_parts = [key for key in where if sorted(p for p in key[: -len(tail)].split(".") if p) == head_parts]
    matches = same_parts if same_parts else list(where)
    if len(matches) != 1:
        raise ValueError(f"cannot find {edit.param_name} in the weight files of {shards[0].parent}: "
                         f"candidates ending in {tail!r}: {sorted(where) or 'none'}")
    return matches[0], where[matches[0]]
