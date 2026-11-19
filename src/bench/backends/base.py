"""The one interface every model backend implements.

The runner (bench.run) and the damage checks (bench.damage) only talk to a model
through these four methods, so a full-precision model, a compressed model and
the dry-run stand-ins are interchangeable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from bench.records import Question


class Backend(ABC):

    @abstractmethod
    def letter_probabilities(self, prompt: str, letters: list[str], question: Question) -> dict[str, float]:
        """Probability of each option letter as the model's very next token after `prompt`.

        `prompt` is the fully formatted prompt text (chat template already applied).
        Returns raw probabilities, not renormalised: {"A": 0.81, "B": 0.05, ...}.
        A letter the model gives (almost) no probability may be reported as 0.0.
        `question` is passed only so the dry-run backends can produce answers;
        real models must ignore it.
        """

    @abstractmethod
    def perplexity(self, texts: list[str]) -> float:
        """Perplexity of the model on these plain texts (no chat template), lower is better.

        Each backend computes it its own way, so values are only comparable between
        runs of the same backend (full vs full, compressed vs compressed).
        """

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> str:
        """Greedy continuation of a plain-text `prompt` (no chat template). Returns only the new text."""

    @abstractmethod
    def describe(self) -> dict:
        """What exactly is being run: backend, model path, weight checksums, library versions."""

    def close(self) -> None:
        """Release the model (free GPU memory, stop a server). Optional."""
