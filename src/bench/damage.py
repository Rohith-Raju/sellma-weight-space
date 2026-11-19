"""Damage checks: does an edit make the model worse at fluent Swedish, even if the exam score holds?

Two measurements, both on text the model has not seen (see bench.damage_data):

1. Perplexity on whole Swedish documents (sv_text.jsonl). Lower is better. A model that has
   lost its grip on the language gives much higher perplexity.
2. Loop rate: give the model 50 short Swedish openings (sv_prompts.jsonl), let it continue each
   one greedily, and count how many continuations get stuck repeating themselves.

Why both: an edit once left a reading-comprehension score unchanged (95.3%) while the model
produced looping nonsense and its perplexity exploded. The exam score alone misses that.

The loop rule is deliberately simple and is written to runs/<id>/damage.json so every result
says which rule produced it. See is_looping below.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from bench.backends.base import Backend
from bench.damage_data import sha256_file
from bench.paths import DAMAGE_DATA
from bench.records import read_jsonl, write_jsonl

MAX_NEW_TOKENS = 200
LOOP_NGRAM = 4           # length of the word sequence to look for, in words
LOOP_MIN_REPEATS = 3     # a reply loops if one such word sequence occurs this many times or more
LOOP_RULE = (
    f"a reply loops if some run of {LOOP_NGRAM} consecutive words (lowercased, punctuation removed) "
    f"occurs {LOOP_MIN_REPEATS} or more times in the reply; greedy continuation, "
    f"max {MAX_NEW_TOKENS} new tokens"
)
TEXT_FILE = "sv_text.jsonl"
PROMPT_FILE = "sv_prompts.jsonl"


def words(text: str) -> list[str]:
    """Lowercased words; punctuation is dropped so 'Writng an "I" (a)' gives writng, an, i, a."""
    return re.findall(r"\w+", text.lower())


def max_ngram_repeats(tokens: list[str], n: int) -> int:
    """How many times the most frequent run of n consecutive tokens occurs (0 if there are fewer than n tokens)."""
    runs = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return max(runs.values(), default=0)


def is_looping(reply: str) -> bool:
    """The loop rule. Flags 'general general general ...' and repeated 'Writng an "I" (a)' patterns.

    Normal prose rarely repeats any four-word run three times in 200 tokens, so ordinary
    answers pass. A false positive makes the loop rate look worse, which is the safe direction.
    """
    return max_ngram_repeats(words(reply), LOOP_NGRAM) >= LOOP_MIN_REPEATS


def load_text_docs(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it with: uv run python -m bench.damage_data")
    return list(read_jsonl(path))


def load_prompts(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; build it with: uv run python -m bench.damage_data")
    return list(read_jsonl(path))


def generate_replies(backend: Backend, prompts: list[dict]) -> list[dict]:
    """One greedy continuation per prompt, each with its loop verdict.

    The full-precision backend can generate all prompts in one batch (about 50x faster). In
    bfloat16 batched and one-by-one texts drift apart after some words because of rounding, so a
    backend always uses the same path: runs of the same backend stay comparable.
    """
    texts = [prompt["prompt"] for prompt in prompts]
    generate_batch = getattr(backend, "generate_batch", None)
    if generate_batch is not None:
        replies = generate_batch(texts, MAX_NEW_TOKENS)
    else:
        replies = [backend.generate(text, max_new_tokens=MAX_NEW_TOKENS) for text in texts]
    return [
        {"id": prompt["id"], "prompt": prompt["prompt"], "reply": reply, "loops": is_looping(reply)}
        for prompt, reply in zip(prompts, replies)
    ]


def run_damage_checks(backend: Backend, run_dir: Path, data_dir: Path = DAMAGE_DATA) -> dict:
    """Measure perplexity and loop rate, write generations.jsonl and damage.json, return the damage dict."""
    texts = load_text_docs(data_dir / TEXT_FILE)
    prompts = load_prompts(data_dir / PROMPT_FILE)
    if not prompts:
        raise ValueError(f"no prompts in {data_dir / PROMPT_FILE}")

    perplexity = backend.perplexity([doc["text"] for doc in texts])
    generations = generate_replies(backend, prompts)
    loop_count = sum(1 for generation in generations if generation["loops"])

    damage = {
        "perplexity": perplexity,
        "n_perplexity_docs": len(texts),
        "loop_rate_percent": 100.0 * loop_count / len(prompts),
        "n_prompts": len(prompts),
        "loop_rule": LOOP_RULE,
        "data_sha256": {
            name: sha256_file(data_dir / name) for name in (TEXT_FILE, PROMPT_FILE)
        },
    }

    write_jsonl(run_dir / "generations.jsonl", generations)
    (run_dir / "damage.json").write_text(json.dumps(damage, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return damage
