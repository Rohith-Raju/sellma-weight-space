"""Tests for the loop rule, the text cleaning, and the damage run itself (offline, with a fake backend)."""

import json
from pathlib import Path

import pytest

from bench.backends.base import Backend
from bench.damage import LOOP_RULE, is_looping, max_ngram_repeats, run_damage_checks, words
from bench.damage_data import (
    MIN_PARAGRAPH_WORDS,
    N_PROMPTS,
    clean_speech_paragraphs,
    clean_wiki_paragraphs,
    prompt_opening,
)
from bench.paths import DAMAGE_DATA
from bench.records import read_jsonl

# Clear loops, taken from the failure modes we actually saw.
LOOPING_REPLIES = [
    "general general general general general general general general general general",
    'Writng an "I" (a) Writng an "I" (a) Writng an "I" (a) Writng an "I" (a) Writng an "I" (a)',
    "Hon gick hem. Hon gick hem. Hon gick hem. Hon gick hem.",
]

# Ordinary Swedish, written for these tests. None of them should be flagged.
NORMAL_REPLIES = [
    "Sverige är ett land i norra Europa med många sjöar och stora skogar.",
    "Idag regnade det hela dagen, så vi stannade hemma och läste en bok om historia.",
    "För att laga soppan behöver du lök, morötter, potatis och lite buljong. "
    "Koka allt i en kastrull i tjugo minuter, och smaka sedan av med salt och peppar.",
    "Riksdagen beslutade i förra veckan om en ny skollag. Debatten var lång, "
    "men de flesta partier var överens om att lärarnas arbetsmiljö måste förbättras.",
    "Att lära sig ett nytt språk tar tid. Man bör läsa varje dag, lyssna på radio "
    "och försöka tala med andra människor, även när det känns svårt. Efter några månader "
    "märker man ofta att meningarna kommer lättare och att ordförrådet växer.",
    "",
    "Ja.",
]


@pytest.mark.parametrize("reply", LOOPING_REPLIES)
def test_loop_rule_flags_clear_loops(reply):
    assert is_looping(reply)


@pytest.mark.parametrize("reply", NORMAL_REPLIES)
def test_loop_rule_passes_normal_swedish(reply):
    assert not is_looping(reply)


def test_words_lowercases_and_drops_punctuation():
    assert words('Writng an "I" (a)') == ["writng", "an", "i", "a"]


def test_max_ngram_repeats_counts_the_most_frequent_run():
    tokens = ["a", "b", "c", "d", "x", "a", "b", "c", "d"]
    assert max_ngram_repeats(tokens, 4) == 2
    assert max_ngram_repeats(["a", "b"], 4) == 0


def test_loop_rule_text_is_recorded_in_the_output():
    assert "3 or more times" in LOOP_RULE


class FakeBackend(Backend):
    """Returns a fixed perplexity and a looping reply for prompts that contain 'LOOP'."""

    def letter_probabilities(self, prompt, letters, question):
        raise NotImplementedError

    def perplexity(self, texts):
        self.seen_texts = list(texts)
        return 12.5

    def generate(self, prompt, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        return "ja ja ja ja ja ja ja ja" if "LOOP" in prompt else "Det var en fin dag i Stockholm."

    def describe(self):
        return {"backend": "fake"}


def write_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    texts = [{"id": "t1", "text": "Ett dokument.\n\nMer text."}, {"id": "t2", "text": "Ett annat dokument."}]
    prompts = [
        {"id": "p1", "prompt": "Det regnade i går."},
        {"id": "p2", "prompt": "LOOP här börjar en text."},
        {"id": "p3", "prompt": "Vi åkte till stan."},
    ]
    (data_dir / "sv_text.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in texts), encoding="utf-8")
    (data_dir / "sv_prompts.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts), encoding="utf-8")


def test_run_damage_checks_writes_generations_and_damage(tmp_path):
    data_dir = tmp_path / "data"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_data(data_dir)
    backend = FakeBackend()

    damage = run_damage_checks(backend, run_dir, data_dir=data_dir)

    assert damage["perplexity"] == 12.5
    assert damage["n_perplexity_docs"] == 2
    assert damage["n_prompts"] == 3
    assert damage["loop_rate_percent"] == pytest.approx(100 / 3)
    assert damage["loop_rule"] == LOOP_RULE
    assert set(damage["data_sha256"]) == {"sv_text.jsonl", "sv_prompts.jsonl"}
    assert backend.seen_texts == ["Ett dokument.\n\nMer text.", "Ett annat dokument."]
    assert backend.max_new_tokens == 200

    generations = list(read_jsonl(run_dir / "generations.jsonl"))
    assert [g["id"] for g in generations] == ["p1", "p2", "p3"]
    assert [g["loops"] for g in generations] == [False, True, False]

    on_disk = json.loads((run_dir / "damage.json").read_text(encoding="utf-8"))
    assert on_disk == damage


def test_run_damage_checks_fails_clearly_without_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="damage_data"):
        run_damage_checks(FakeBackend(), tmp_path, data_dir=tmp_path / "missing")


# ---------------------------------------------------------------- cleaning

def test_wiki_cleaning_drops_headings_and_everything_after_references():
    extract = "\n".join([
        "Inledning med en mening som är tillräckligt lång för att räknas som prosa här.",
        "",
        "== Historia ==",
        "En andra paragraf med tillräckligt många ord för att tas med i texten också.",
        "* En punkt i en lista som inte är prosa men kanske något längre än annars.",
        "== Referenser ==",
        "Den här raden hör till bibliografin och ska inte finnas med i texten alls.",
    ])
    paragraphs = clean_wiki_paragraphs(extract)
    assert len(paragraphs) == 2
    assert all(len(p.split()) >= MIN_PARAGRAPH_WORDS for p in paragraphs)
    assert not any("bibliografin" in p for p in paragraphs)


def test_speech_cleaning_drops_stage_directions_and_html():
    speech = (
        "<p>Herr talman! Jag vill säga något om skolan och om lärarnas villkor i dag.</p>"
        "<p>(Applåder)</p><p>Tack.</p>"
        "<p>Det handlar om &quot;kvalitet&quot; och om att alla elever ska få en bra start.</p>"
    )
    paragraphs = clean_speech_paragraphs(speech)
    assert paragraphs[0].startswith("Herr talman!")
    assert paragraphs[1].startswith('Det handlar om "kvalitet"')
    assert len(paragraphs) == 2


def test_prompt_opening_takes_one_to_three_sentences_of_30_to_60_words():
    long_sentence = " ".join(["ord"] * 45) + "."
    assert prompt_opening([long_sentence + " Mer text här."]) == long_sentence
    too_long = " ".join(["ord"] * 70) + "."
    assert prompt_opening([too_long]) is None


# ---------------------------------------------------------------- the built data

@pytest.mark.skipif(not (DAMAGE_DATA / "sv_text.jsonl").exists(), reason="data/damage not built yet")
def test_built_data_meets_the_rules():
    texts = list(read_jsonl(DAMAGE_DATA / "sv_text.jsonl"))
    prompts = list(read_jsonl(DAMAGE_DATA / "sv_prompts.jsonl"))

    assert 60 <= len(texts) <= 150
    assert sum(len(t["text"].split()) for t in texts) >= 40_000
    assert len(prompts) == N_PROMPTS
    for doc in texts + prompts:
        assert doc["published"] > "2025-05-01", doc["id"]
        assert doc["url"].startswith("https://"), doc["id"]
    for prompt in prompts:
        assert 30 <= len(prompt["prompt"].split()) <= 60, prompt["id"]
    assert not {t["id"] for t in texts} & {p["id"] for p in prompts}
