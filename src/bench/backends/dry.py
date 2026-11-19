"""Stand-in backends that need no model. They exist to test the pipeline end to end.

RandomBackend answers at random, but the same way every time for the same seed and
question, so a dry run is repeatable. Its accuracy should land near chance (about 20-25%).

OracleBackend always puts most of its probability on the answer key. Its accuracy must be
100%; if it is not, the scoring code is wrong.

Neither backend reads the prompt for letter probabilities. Their generate() and
perplexity() return fixed-shape made-up values, so they are only useful for plumbing.
"""

from __future__ import annotations

import hashlib

import numpy as np

from bench.backends.base import Backend
from bench.records import Question

RANDOM_LETTER_MASS = 0.9     # the random backend leaves 10% of the probability on other tokens
ORACLE_CORRECT_PROB = 0.97   # the oracle puts this much on the answer key, the rest is spread out


class RandomBackend(Backend):

    def __init__(self, seed: int):
        self.seed = seed

    def letter_probabilities(self, prompt: str, letters: list[str], question: Question) -> dict[str, float]:
        rng = np.random.default_rng(_stable_seed(self.seed, question.id))
        weights = rng.random(len(letters))
        probs = RANDOM_LETTER_MASS * weights / weights.sum()
        return {letter: float(p) for letter, p in zip(letters, probs)}

    def perplexity(self, texts: list[str]) -> float:
        # A plausible-looking Swedish-text perplexity, fixed per seed.
        rng = np.random.default_rng(_stable_seed(self.seed, "perplexity"))
        return float(rng.uniform(15.0, 30.0))

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return _pseudo_text(prompt, max_new_tokens, salt=f"random-{self.seed}")

    def describe(self) -> dict:
        return {"backend": "random", "path": None, "seed": self.seed}


class OracleBackend(Backend):

    def letter_probabilities(self, prompt: str, letters: list[str], question: Question) -> dict[str, float]:
        if question.answer not in letters:
            # Should not happen with real data; a uniform answer keeps the run going and shows as wrong.
            return {letter: 1.0 / len(letters) for letter in letters}
        others = [letter for letter in letters if letter != question.answer]
        probs = {letter: 0.0 for letter in letters}
        probs[question.answer] = ORACLE_CORRECT_PROB
        for letter in others:
            probs[letter] = (1.0 - ORACLE_CORRECT_PROB) / len(others)
        return probs

    def perplexity(self, texts: list[str]) -> float:
        return 12.0

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return _pseudo_text(prompt, max_new_tokens, salt="oracle")

    def describe(self) -> dict:
        return {"backend": "oracle", "path": None}


class BrokenBackend(OracleBackend):
    """Imitates the damaged edit seen earlier: perfect exam answers, but the model loops when it
    writes and its perplexity has exploded. Comparing it against the oracle must fail the damage
    rule even though the exam score is identical; if it does not, the damage checks are broken."""

    def perplexity(self, texts: list[str]) -> float:
        return 5_000_000.0

    def generate(self, prompt: str, max_new_tokens: int) -> str:
        return " ".join(["general"] * min(max_new_tokens, 60))

    def describe(self) -> dict:
        return {"backend": "broken", "path": None}


def _stable_seed(seed: int, key: str) -> int:
    """A seed that does not change between Python runs (the built-in hash() is salted)."""
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _pseudo_text(prompt: str, max_new_tokens: int, salt: str) -> str:
    """Reuses words from the prompt in a shuffled order. Every word appears at most once, so the
    output cannot loop, and the same prompt always gives the same text."""
    words = list(dict.fromkeys(prompt.split()))
    if not words or max_new_tokens <= 0:
        return ""
    rng = np.random.default_rng(_stable_seed(0, f"{salt}|{prompt}"))
    count = min(max_new_tokens, len(words))
    picked = [words[i] for i in rng.permutation(len(words))[:count]]
    return " ".join(picked)
