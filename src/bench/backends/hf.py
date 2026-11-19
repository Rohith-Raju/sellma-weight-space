"""Full-precision model through Hugging Face transformers, on a CUDA GPU.

Number format: bfloat16 by default, the format the weights are published in. `dtype = "float32"`
in config/models.toml computes with the same weights in float32 instead. That is twice the
memory and up to twice the time on long texts, but more exact: Gemma's answer logits are around
50, where bfloat16 can only tell values 0.25 apart, and a small weight edit (smaller than about
0.4% of a weight) is rounded away entirely in bfloat16. Runs are only comparable with runs of
the same dtype; it is recorded in run.json.

Scoring reads the model's next-token probabilities after the prompt, so a question costs
one forward pass and no sampling. The probability of an option letter is the sum over
every vocabulary token that is that letter once surrounding whitespace is stripped
("A" and " A" both count). Those token sets are found once, when the backend starts.

Perplexity is the mean per-token negative log-likelihood over all texts, exponentiated.
Texts longer than WINDOW tokens are cut into consecutive non-overlapping windows; the
first token of each window is then not scored, which is a negligible effect at this length.

Loading: the checkpoint is loaded as the architecture named in its config.json. Both models
are published as image+text checkpoints (Qwen3_5ForConditionalGeneration,
Gemma3ForConditionalGeneration). Loading that class, and feeding it text only, gives exactly the
same logits as the text-only class, and it keeps every parameter name equal to the name in the
safetensors files. The edit code relies on that: it changes weights through `backend.model`
and saves them back under the same names. The language model's blocks are at
`backend.model.model.language_model.layers[i]` for both models.

Qwen3.5's linear-attention layers run on transformers' built-in PyTorch code. The optional
faster kernels (flash-linear-attention, causal-conv1d) are not installed: the built-in path is
the reference implementation and is fast enough here (about 0.1 s per question).

Free generation is batched (generate_batch): the 50 damage-check prompts are continued together,
left-padded so every prompt ends at the last position. One at a time takes minutes per run.

torch and transformers are imported in __init__, so importing this module is cheap and
needs no GPU libraries.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from bench.backends import DEFAULT_HF_DTYPE, folder_checksums
from bench.backends.base import Backend
from bench.records import LETTERS, Question

WINDOW = 1024   # longest sequence scored at once; bounds the logits memory (about 1 GB at this length)
GENERATE_BATCH_SIZE = 64   # prompts continued together; the 50 damage prompts fit in one batch


class HFBackend(Backend):

    def __init__(self, model_path: Path, tokenizer_path: Path, window: int = WINDOW, dtype: str = DEFAULT_HF_DTYPE):
        import torch
        import transformers
        from transformers import AutoTokenizer

        self._torch = torch
        self._transformers_version = transformers.__version__
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.window = window
        self.dtype = dtype
        # Public on purpose: the edit code changes weights in memory through `model`
        # and tokenises its own prompts with `tokenizer`.
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        model_class = _model_class(self.model_path)
        self.model = model_class.from_pretrained(str(self.model_path), dtype=getattr(torch, dtype), device_map="cuda")
        self.model.eval()
        self._model_class_name = model_class.__name__
        self._letter_ids = _letter_token_ids(self.tokenizer, LETTERS)

    def letter_probabilities(self, prompt: str, letters: list[str], question: Question) -> dict[str, float]:
        torch = self._torch
        # add_special_tokens=False: the chat template already writes BOS where the model needs it.
        input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids
        with torch.inference_mode():
            # Only the last position is needed, which saves computing logits for the whole prompt.
            logits = self.model(input_ids=input_ids.to(self.model.device), logits_to_keep=1).logits[0, -1]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return {letter: _mass(probs, self._letter_ids[letter]) for letter in letters}

    def perplexity(self, texts: list[str]) -> float:
        total_nll = 0.0
        total_scored = 0
        for text in texts:
            # add_special_tokens=True: the tokenizer adds BOS where the model expects it.
            ids = self.tokenizer(text, add_special_tokens=True).input_ids
            for start in range(0, len(ids), self.window):
                chunk = ids[start:start + self.window]
                if len(chunk) < 2:
                    continue
                nll, scored = self._window_nll(chunk)
                total_nll += nll
                total_scored += scored
        if total_scored == 0:
            raise ValueError("no tokens to score for perplexity")
        return math.exp(total_nll / total_scored)

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return self.generate_batch([prompt], max_new_tokens)[0]

    def generate_batch(self, prompts: list[str], max_new_tokens: int,
                       batch_size: int = GENERATE_BATCH_SIZE) -> list[str]:
        """Greedy continuations of several plain-text prompts at once, in the order given.

        Much faster than one prompt at a time. The result for a prompt can differ in rare
        details from generating it alone (padding changes the arithmetic in its last bits), so
        runs that are compared with each other must both use this method with the same prompts.
        """
        replies: list[str] = []
        for start in range(0, len(prompts), batch_size):
            replies.extend(self._generate_padded(prompts[start:start + batch_size], max_new_tokens))
        return replies

    def _generate_padded(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        torch = self._torch
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        # Left padding: every prompt must end at the last position, where generation continues.
        encoded = self.tokenizer(prompts, add_special_tokens=True, return_tensors="pt",
                                 padding=True, padding_side="left").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=encoded.input_ids,
                attention_mask=encoded.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy: the same prompts always give the same texts
                pad_token_id=pad_id,
            )
        new_ids = output[:, encoded.input_ids.shape[1]:]
        return self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)

    def describe(self) -> dict:
        torch = self._torch
        description = {
            "backend": "hf",
            "path": str(self.model_path),
            "tokenizer": str(self.tokenizer_path),
            "model_class": self._model_class_name,
            "model_files": folder_checksums(self.model_path),
            "dtype": self.dtype,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": self._transformers_version,
        }
        if self.tokenizer_path.resolve() != self.model_path.resolve():
            description["tokenizer_files"] = folder_checksums(self.tokenizer_path)
        return description

    def close(self) -> None:
        self.model = None
        self._torch.cuda.empty_cache()

    def _window_nll(self, ids: list[int]) -> tuple[float, int]:
        """Summed negative log-likelihood of the tokens after the first one, and how many there are."""
        torch = self._torch
        input_ids = torch.tensor([ids], device=self.model.device)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits[0, :-1].float()
            targets = input_ids[0, 1:]
            nll = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
        return float(nll.item()), len(ids) - 1


def _model_class(model_path: Path):
    """The transformers class named in the checkpoint's config.json (see the module docstring for why).

    Falls back to AutoModelForCausalLM for a checkpoint that names no class transformers knows.
    """
    import transformers

    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    for name in config.get("architectures") or []:
        model_class = getattr(transformers, name, None)
        if model_class is not None:
            return model_class
    return transformers.AutoModelForCausalLM


def _letter_token_ids(tokenizer, letters: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Every vocabulary id whose decoded text, stripped of whitespace, is exactly one letter."""
    found: dict[str, list[int]] = {letter: [] for letter in letters}
    for token_id in range(len(tokenizer)):
        text = tokenizer.decode([token_id]).strip()
        if text in found:
            found[text].append(token_id)
    return {letter: np.array(ids, dtype=np.int64) for letter, ids in found.items()}


def _mass(probs: np.ndarray, token_ids: np.ndarray) -> float:
    # Ids beyond the model's output size (tokenizer padding) have no probability.
    in_range = token_ids[token_ids < len(probs)]
    return float(probs[in_range].sum())
