"""Turns one exam question into the exact text a model is asked.

Two steps, so that every backend sees the same text:

1. build_prompt(question) writes the chat messages: Swedish instructions modelled on
   the exam's own, then the passage or sentence, then the options as "A. ..." lines,
   then a line asking for the letter only. Nothing model-specific happens here.
2. render_prompt(messages, tokenizer_path) applies the model's chat template from
   its Hugging Face tokenizer folder. This is the only model-specific step, and both
   the full-precision and the compressed copy of a model point at the same folder,
   so they receive byte-identical prompt strings.

Thinking models (Qwen3.5) are asked with enable_thinking=False. The template then
writes an empty <think></think> block after the assistant tag, so the first token the
model generates is the answer letter. Gemma's template has no such switch.
"""

from __future__ import annotations

import functools
from typing import Any

from bench.records import Question

# Instructions in the same register as the exam booklets. Kept short and neutral on purpose.
ORD_INSTRUCTION = "Välj det alternativ som har närmast samma betydelse som ordet eller uttrycket."
LAS_INSTRUCTION = "Läs texten och svara på frågan. Välj det alternativ som bäst stämmer med texten."
MEK_INSTRUCTION = (
    "Välj det alternativ som bäst passar in i luckorna (_____) i meningen. "
    "Om alternativet består av flera ord gäller ordningen från vänster till höger i meningen."
)


def build_prompt(question: Question) -> list[dict[str, str]]:
    """Chat messages for one question: a single user message, no system message."""
    body = "\n\n".join(_section_parts(question) + [_options_block(question), _answer_line(question)])
    return [{"role": "user", "content": body}]


def _section_parts(question: Question) -> list[str]:
    """The instruction and question text, which differs by section. The options come after."""
    if question.section == "ORD":
        return [ORD_INSTRUCTION, f"Ord eller uttryck: {question.question.strip()}"]
    if question.section == "LAS":
        return [LAS_INSTRUCTION, *_passage_parts(question), f"Fråga: {question.question.strip()}"]
    if question.section == "MEK":
        return [MEK_INSTRUCTION, f"Mening: {question.question.strip()}"]
    raise ValueError(f"unknown section {question.section!r} in question {question.id}")


def _passage_parts(question: Question) -> list[str]:
    """LAS passage with its title. Titles are optional in the data, so skip them when absent."""
    parts = []
    if question.passage_title:
        parts.append(f"Titel: {question.passage_title.strip()}")
    parts.append((question.passage or "").strip())
    return parts


def _options_block(question: Question) -> str:
    # Sorted so the letter order is always A, B, C, ... whatever order the data was stored in.
    return "\n".join(f"{letter}. {question.options[letter].strip()}" for letter in sorted(question.options))


def _answer_line(question: Question) -> str:
    letters = sorted(question.options)
    return f"Svara endast med bokstaven för det alternativ du väljer ({_swedish_or_list(letters)})."


def _swedish_or_list(letters: list[str]) -> str:
    """["A", "B", "C"] -> "A, B eller C"."""
    if len(letters) < 2:
        return "".join(letters)
    return f"{', '.join(letters[:-1])} eller {letters[-1]}"


def render_prompt(messages: list[dict[str, Any]], tokenizer_path: str | None) -> str:
    """The prompt string the model is fed, with the chat template applied.

    tokenizer_path=None is used by dry runs, which have no template: a plain role-labelled
    text is returned instead.
    """
    if tokenizer_path is None:
        return _plain_rendering(messages)
    tokenizer = _load_tokenizer(tokenizer_path)
    # add_generation_prompt opens the assistant turn; enable_thinking=False closes an empty
    # think block for Qwen. Other templates ignore the extra variable.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _plain_rendering(messages: list[dict[str, Any]]) -> str:
    lines = [f"{message['role']}: {message['content']}" for message in messages]
    lines.append("assistant:")
    return "\n\n".join(lines)


@functools.lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_path: str):
    # Imported here so that importing this module (and the dry runs) does not need transformers.
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)
