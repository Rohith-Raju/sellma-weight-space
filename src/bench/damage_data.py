"""Build the Swedish texts used by the damage checks (data/damage/).

Two files come out of this:

* sv_text.jsonl: whole documents. The bench measures perplexity on these.
* sv_prompts.jsonl: short openings of other documents. The bench gives each opening to the
  model as a prompt and checks whether the continuation gets stuck in a loop.

Why these texts: an edit can keep the exam scores and still wreck the model's fluent Swedish,
so the damage checks need text the model has not seen. Both models have training cutoffs before
2025-05-01 (Gemma 3: August 2024; Qwen3.5: April 2025), so every document here was first
published after that date.

Sources (both openly available; SOURCES.md records the licences):
* Swedish Wikipedia articles created after the cutoff, from the page-creation log (CC BY-SA 4.0).
* Speeches (anföranden) in the Riksdag's 2025/26 session, from Riksdagen open data.

Selection is deterministic. Each source's candidates are put in a fixed pseudo-random order
(sorted by the SHA-256 of their id, so the order does not follow dates or titles). The two
orders are interleaved, the first usable documents become prompts, and the rest fill the
perplexity text up to about 40,000 words. Rebuilding gives the same files unless the upstream
sources have changed; SOURCES.md records the fetch date and the checksums of the output.

Run with:  uv run python -m bench.damage_data
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from urllib.parse import quote

import requests

from bench.paths import DAMAGE_DATA
from bench.records import write_jsonl

USER_AGENT = "selma-bench/0.1 (research)"
CUTOFF = "2025-05-01"            # documents must be published strictly after this (ISO dates compare as text)

WIKI_API = "https://sv.wikipedia.org/w/api.php"
WIKI_LICENSE = "CC BY-SA 4.0"
RIKSDAGEN_SESSION = "2025/26"
SPEECH_DUMP_URL = "https://data.riksdagen.se/dataset/anforande/anforande-202526.json.zip"   # every speech of the session
RIKSDAGEN_SPEECH_PAGE_URL = "https://data.riksdagen.se/anforande/{speech_id}"              # citation link per speech
RIKSDAGEN_LICENSE = "Public parliamentary record, Riksdagen open data"

CACHE_DIR = Path.home() / ".cache" / "selma-bench" / "damage_raw"   # raw API responses, outside the repo
SPEECH_DUMP_PATH = CACHE_DIR.parent / "anforande-202526.json.zip"
REQUEST_DELAY_S = 1.0            # pause before every request, to be polite to the servers
MAX_ATTEMPTS = 6
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_CREATION_PAGES = 20         # 500 creations per page; caps the Wikipedia log walk (it is throttled)

TEXT_WORD_TARGET = 40_000        # total words of perplexity text
MAX_TEXT_DOCS = 150
N_PROMPTS = 50
MIN_DOC_WORDS = 150              # shorter documents are too thin for a perplexity sample
POOL_SIZE = 160                  # usable documents to collect per source before choosing
WIKI_BATCH = 20                  # the extracts API returns at most 20 plain-text extracts per call
PROMPT_MIN_WORDS = 30
PROMPT_MAX_WORDS = 60
PROMPT_MAX_SENTENCES = 3
MIN_PARAGRAPH_WORDS = 8

# Sections of a Wikipedia article that are bibliography, not prose. Everything after them is dropped.
REFERENCE_HEADINGS = {
    "referenser", "källor", "noter", "litteratur", "externa länkar", "se även",
    "vidare läsning", "bibliografi", "källförteckning", "fotnoter", "externa länkar och källor",
}
HEADING_RE = re.compile(r"^(={2,})\s*(.+?)\s*\1$")
# Splits after . ! or ? when a capital letter, digit or quote follows, so "t.ex. och" stays whole.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ0-9\"“«(])")
SENTENCE_END_CHARS = (".", "!", "?", "»", "”", '"', ")")
NON_PROSE_START_CHARS = "*#|;:=("


@dataclass
class Candidate:
    """One usable document, cleaned to prose paragraphs, before the final selection."""
    doc_id: str
    source: str
    url: str
    published: str           # ISO date of first publication
    title: str
    license: str
    paragraphs: list[str]

    @property
    def word_count(self) -> int:
        return sum(len(paragraph.split()) for paragraph in self.paragraphs)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


def log(message: str) -> None:
    """Progress goes to stderr, so the data files on stdout stay clean."""
    print(message, file=sys.stderr, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_order_key(value: str) -> str:
    """A fixed pseudo-random order: the same ids always sort the same way, unrelated to dates."""
    return sha256_bytes(value.encode("utf-8"))


# ---------------------------------------------------------------- text cleaning

def is_prose(line: str) -> bool:
    """True for a line that reads as a sentence or paragraph, not a list, table, heading or stage direction."""
    if len(line.split()) < MIN_PARAGRAPH_WORDS:
        return False
    if line[0] in NON_PROSE_START_CHARS or "|" in line:
        return False
    return line.endswith(SENTENCE_END_CHARS)


def clean_wiki_paragraphs(extract: str) -> list[str]:
    """Plain-text Wikipedia extract -> prose paragraphs. Stops at the first reference section."""
    paragraphs: list[str] = []
    for raw_line in extract.split("\n"):
        line = raw_line.strip()
        heading = HEADING_RE.match(line)
        if heading:
            if heading.group(2).lower() in REFERENCE_HEADINGS:
                break  # the rest of the article is bibliography
            continue
        if is_prose(line):
            paragraphs.append(line)
    return paragraphs


def clean_speech_paragraphs(speech_html: str) -> list[str]:
    """Riksdagen speech HTML (one <p> per paragraph) -> prose paragraphs.

    Stage directions such as "(Applåder)" start with "(" and are dropped by is_prose.
    """
    paragraphs: list[str] = []
    for chunk in speech_html.split("</p>"):
        text = html.unescape(re.sub(r"<[^>]+>", " ", chunk))
        text = " ".join(text.split())
        if text and is_prose(text):
            paragraphs.append(text)
    return paragraphs


def prompt_opening(paragraphs: list[str]) -> str | None:
    """The first 1-3 sentences that add up to 30-60 words, or None if the document has no such opening."""
    chosen: list[str] = []
    words = 0
    for paragraph in paragraphs:
        for sentence in SENTENCE_SPLIT_RE.split(paragraph):
            count = len(sentence.split())
            if words + count > PROMPT_MAX_WORDS or len(chosen) == PROMPT_MAX_SENTENCES:
                return None
            chosen.append(sentence)
            words += count
            if words >= PROMPT_MIN_WORDS:
                return " ".join(chosen)
    return None


# ---------------------------------------------------------------- HTTP

def cache_path(url: str, params: dict[str, str]) -> Path:
    key = sha256_bytes(f"{url}?{sorted(params.items())}".encode("utf-8"))
    return CACHE_DIR / f"{key}.json"


def get_json(session: requests.Session, url: str, params: dict[str, str]) -> dict:
    """GET a JSON document, politely paced. Rate limits and server errors are retried with a longer wait.

    The wait honours Retry-After when the server sends it, because hammering a server that
    says "too many requests" only makes it refuse us for longer.
    """
    cached = cache_path(url, params)
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    log(f"request {url.split('/')[2]} {params.get('letype', '') or params.get('p', '') or params.get('pageids', '')[:30]}")
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        time.sleep(REQUEST_DELAY_S)
        try:
            response = session.get(url, params=params, timeout=60)
            if response.status_code in RETRY_STATUS_CODES:
                log(f"  HTTP {response.status_code} (attempt {attempt}), Retry-After {response.headers.get('Retry-After')}")
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            data = response.json()
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data
        except requests.RequestException as error:
            last_error = error
            time.sleep(retry_wait_seconds(error, attempt))
    raise RuntimeError(f"giving up on {url} after {MAX_ATTEMPTS} attempts") from last_error


def retry_wait_seconds(error: requests.RequestException, attempt: int) -> float:
    response = getattr(error, "response", None)
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after and retry_after.isdigit():
        return float(retry_after)
    return 15.0 * attempt


# ---------------------------------------------------------------- Swedish Wikipedia

def wiki_url(title: str) -> str:
    return "https://sv.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))


def wiki_creations(session: requests.Session) -> list[dict]:
    """Every main-namespace article whose creation is logged after CUTOFF, oldest first."""
    base = {
        "action": "query", "list": "logevents", "letype": "create", "lenamespace": "0",
        "lestart": "2025-05-02T00:00:00Z", "ledir": "newer", "lelimit": "500", "format": "json",
        "maxlag": "5",  # asks Wikimedia to refuse us while its servers are overloaded
    }
    rows: list[dict] = []
    seen: set[int] = set()
    pages_read = 0
    params = dict(base)
    while True:
        data = get_json(session, WIKI_API, params)
        for event in data["query"]["logevents"]:
            if event["pageid"] not in seen:
                seen.add(event["pageid"])
                rows.append({"pageid": event["pageid"], "title": event["title"], "timestamp": event["timestamp"]})
        pages_read += 1
        if "continue" not in data or pages_read == MAX_CREATION_PAGES:
            return rows
        params = {**base, **data["continue"]}


def wiki_extracts(session: requests.Session, pageids: list[int]) -> dict[str, dict]:
    """Plain-text extracts (no templates, no infoboxes) keyed by page id as a string."""
    data = get_json(session, WIKI_API, {
        "action": "query", "prop": "extracts", "explaintext": "1", "exlimit": str(WIKI_BATCH),
        "pageids": "|".join(str(pageid) for pageid in pageids), "format": "json", "maxlag": "5",
    })
    return data.get("query", {}).get("pages", {})


def wiki_candidate(row: dict, page: dict | None) -> Candidate | None:
    if page is None or "missing" in page:
        return None  # deleted after it was created
    return Candidate(
        doc_id=f"sv-wiki-{row['pageid']}",
        source="sv.wikipedia.org",
        url=wiki_url(page["title"]),
        published=row["timestamp"][:10],
        title=page["title"],
        license=WIKI_LICENSE,
        paragraphs=clean_wiki_paragraphs(page.get("extract", "")),
    )


def collect_wiki_pool(session: requests.Session) -> list[Candidate]:
    """Usable Wikipedia articles, in fixed pseudo-random order, until POOL_SIZE is reached."""
    creations = sorted(wiki_creations(session), key=lambda row: stable_order_key(str(row["pageid"])))
    pool: list[Candidate] = []
    for start in range(0, len(creations), WIKI_BATCH):
        batch = creations[start:start + WIKI_BATCH]
        try:
            pages = wiki_extracts(session, [row["pageid"] for row in batch])
        except RuntimeError as error:
            # Wikimedia throttles this client for long stretches. Stop here and build from what we have;
            # SOURCES.md records the sources actually used, so a short Wikipedia pool is visible there.
            log(f"wiki: giving up after throttling ({error}); continuing with {len(pool)} articles")
            break
        if not any(page.get("extract") for page in pages.values()):
            # Answered, but with no text at all: the API is refusing us, not the articles being empty.
            log(f"wiki: a batch returned no text; stopping the Wikipedia stage with {len(pool)} articles")
            break
        log(f"wiki: checked {start + len(batch)}/{len(creations)} creations, {len(pool)} usable")
        for row in batch:
            candidate = wiki_candidate(row, pages.get(str(row["pageid"])))
            if candidate and candidate.word_count >= MIN_DOC_WORDS and candidate.published > CUTOFF:
                pool.append(candidate)
        if len(pool) >= POOL_SIZE:
            break
    return pool


# ---------------------------------------------------------------- Riksdagen speeches

def download_speech_dump(session: requests.Session) -> Path:
    """The bulk file with every speech of the session (about 24 MB), downloaded once and cached.

    One download replaces thousands of per-speech API requests. The listing API also ignores
    page numbers, so the bulk file is the only reliable way to see every speech.
    """
    if not SPEECH_DUMP_PATH.exists():
        log(f"downloading {SPEECH_DUMP_URL}")
        response = session.get(SPEECH_DUMP_URL, timeout=300)
        response.raise_for_status()
        SPEECH_DUMP_PATH.parent.mkdir(parents=True, exist_ok=True)
        partial = SPEECH_DUMP_PATH.with_suffix(".partial")
        partial.write_bytes(response.content)
        partial.rename(SPEECH_DUMP_PATH)
    return SPEECH_DUMP_PATH


def read_speech_dump(dump: Path) -> list[dict]:
    """Every speech in the bulk file. Each member is one JSON file with a UTF-8 byte-order mark."""
    speeches: list[dict] = []
    with zipfile.ZipFile(dump) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith(".json"):
                speeches.append(json.loads(archive.read(name).decode("utf-8-sig"))["anforande"])
    return speeches


def speech_candidate(row: dict) -> Candidate:
    speech_id = f"{row['dok_id']}-{row['anforande_nummer']}"  # e.g. HD09157-121
    return Candidate(
        doc_id=f"riksdagen-{row['anforande_id']}",
        source="data.riksdagen.se",
        url=RIKSDAGEN_SPEECH_PAGE_URL.format(speech_id=speech_id),
        published=row["dok_datum"][:10],
        title=f"{row['talare']}, {row['dok_titel']}: {row['avsnittsrubrik']}".strip(", :"),
        license=RIKSDAGEN_LICENSE,
        paragraphs=clean_speech_paragraphs(row["anforandetext"] or ""),
    )


def collect_speech_pool(session: requests.Session) -> list[Candidate]:
    """Usable speeches (not replies, after CUTOFF, long enough), in fixed pseudo-random order."""
    rows = [row for row in read_speech_dump(download_speech_dump(session))
            if row["replik"] == "N" and row["dok_datum"][:10] > CUTOFF]
    rows.sort(key=lambda row: stable_order_key(row["anforande_id"]))
    pool: list[Candidate] = []
    for row in rows:
        candidate = speech_candidate(row)
        if candidate.word_count >= MIN_DOC_WORDS:
            pool.append(candidate)
        if len(pool) >= POOL_SIZE:
            break
    log(f"speeches: {len(rows)} in the bulk file after the cutoff, {len(pool)} taken")
    return pool


# ---------------------------------------------------------------- selection

def interleave(first: list[Candidate], second: list[Candidate]) -> list[Candidate]:
    """Alternate between the two sources, so the texts are split evenly between them."""
    merged: list[Candidate] = []
    for a, b in zip_longest(first, second):
        if a is not None:
            merged.append(a)
        if b is not None:
            merged.append(b)
    return merged


def choose_prompts(ordered: list[Candidate]) -> tuple[list[tuple[Candidate, str]], list[Candidate]]:
    """The first N_PROMPTS documents with a usable opening become prompts. Returns (prompts, the rest)."""
    prompts: list[tuple[Candidate, str]] = []
    used: set[str] = set()
    for candidate in ordered:
        if len(prompts) == N_PROMPTS:
            break
        opening = prompt_opening(candidate.paragraphs)
        if opening is not None:
            prompts.append((candidate, opening))
            used.add(candidate.doc_id)
    rest = [candidate for candidate in ordered if candidate.doc_id not in used]
    return prompts, rest


def choose_text(rest: list[Candidate]) -> list[Candidate]:
    """Whole documents from the rest, in order, until the word target is reached."""
    chosen: list[Candidate] = []
    words = 0
    for candidate in rest:
        if words >= TEXT_WORD_TARGET or len(chosen) == MAX_TEXT_DOCS:
            break
        chosen.append(candidate)
        words += candidate.word_count
    return chosen


# ---------------------------------------------------------------- output

def text_record(candidate: Candidate) -> dict:
    return {
        "id": candidate.doc_id, "source": candidate.source, "url": candidate.url,
        "title": candidate.title, "published": candidate.published,
        "license": candidate.license, "text": candidate.text,
    }


def prompt_record(candidate: Candidate, opening: str) -> dict:
    return {
        "id": candidate.doc_id, "source": candidate.source, "url": candidate.url,
        "published": candidate.published, "prompt": opening,
    }


def source_summary(candidates: list[Candidate]) -> dict:
    """Counts per source, for SOURCES.md."""
    summary: dict[str, dict] = {}
    for candidate in candidates:
        entry = summary.setdefault(candidate.source, {
            "docs": 0, "words": 0, "first": candidate.published, "last": candidate.published,
            "license": candidate.license,
        })
        entry["docs"] += 1
        entry["words"] += candidate.word_count
        entry["first"] = min(entry["first"], candidate.published)
        entry["last"] = max(entry["last"], candidate.published)
    return summary


def render_sources_md(fetched: str, text_docs: list[Candidate], prompts: list[tuple[Candidate, str]],
                      checksums: dict[str, str]) -> str:
    text_summary = source_summary(text_docs)
    total_words = sum(c.word_count for c in text_docs)
    lines = [
        "# Swedish text for the damage checks",
        "",
        f"Built by `bench.damage_data` on {fetched} (fetch date). Rebuild with `uv run python -m bench.damage_data`.",
        "",
        "## What was collected",
        "",
        f"* `sv_text.jsonl`: {len(text_docs)} documents, {total_words} words, for perplexity.",
        f"* `sv_prompts.jsonl`: {len(prompts)} prompts (the opening 30-60 words, 1-3 sentences) of other documents, "
        "not in `sv_text.jsonl`.",
        "",
        "| Source | Documents (text) | Words (text) | Published | Licence |",
        "|---|---|---|---|---|",
    ]
    for source, entry in sorted(text_summary.items()):
        lines.append(f"| {source} | {entry['docs']} | {entry['words']} | "
                     f"{entry['first']} to {entry['last']} | {entry['license']} |")
    lines += [
        "",
        f"Every document was first published after {CUTOFF}, after the training cutoffs of Gemma 3 (August 2024) "
        "and Qwen3.5 (April 2025).",
        "",
        "## Sources and attribution",
        "",
        "* **Swedish Wikipedia** (sv.wikipedia.org). Articles whose page creation is logged after the cutoff, "
        "text from the plain-text extract API. Licence CC BY-SA 4.0: credit the article title and URL "
        "(as listed per document in the JSONL files), and share adaptations under the same licence. "
        "The text is the article's current version when fetched, which may be newer than its creation.",
        "* **Riksdagen open data** (data.riksdagen.se). Speeches (anföranden) in the 2025/26 session, read from "
        f"the bulk file {SPEECH_DUMP_URL} (sha256 `{sha256_file(SPEECH_DUMP_PATH)}`). Public parliamentary record. "
        "Cite the speaker, debate and riksdagen.se URL. "
        "Terms of use (Riksdagen's \"Användningsvillkor\", "
        "https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anvandarstod/anvandningsvillkor/, "
        "checked 2026-11-19): the open data may be used and redistributed freely as long as the source is given, "
        "as \"Källa: Sveriges riksdag\" or \"Source: The Swedish Parliament\". Do not suggest that the Riksdag is "
        "behind or endorses the work. The data contains personal data (the speakers' names and speeches), so the "
        "GDPR and lag (2018:218) apply to any further use.",
        "",
        "Attribution for the Riksdagen text in this folder (and the prompts repeated in the `generations.jsonl` "
        "files of the run folders):",
        "",
        "> Källa: Sveriges riksdag / Source: The Swedish Parliament",
        "",
        "## Selection and cleaning",
        "",
        "* Candidates per source are sorted by SHA-256 of their id (a fixed, date-independent order), then the two "
        "sources are interleaved one by one.",
        f"* The first {N_PROMPTS} documents with a usable opening (30-60 words, 1-3 sentences) become prompts. "
        "They are excluded from the text.",
        f"* The rest, in the same order, fill the text until about {TEXT_WORD_TARGET} words "
        f"(at most {MAX_TEXT_DOCS} documents). Each text document has at least {MIN_DOC_WORDS} words.",
        "* Cleaning keeps prose paragraphs of at least 8 words that end with sentence punctuation. Drops headings, "
        "lists, tables, reference sections (everything after Referenser, Källor, Noter, Se även and similar), "
        "and stage directions such as (Applåder).",
        "",
        "## Checksums",
        "",
    ]
    for name, digest in sorted(checksums.items()):
        lines.append(f"* `{name}`: `{digest}`")
    lines += [
        "",
        "## Caveats",
        "",
        "* Upstream text can be edited after the fetch, so a rebuild on another date can differ. "
        "Compare the checksums above to see whether the files are the same.",
        "* The sentence splitter is simple; a few prompts may end at an abbreviation or start mid-thought.",
        "",
    ]
    return "\n".join(lines)


def build_damage_data(out_dir: Path = DAMAGE_DATA) -> dict:
    """Fetch, select and write sv_text.jsonl, sv_prompts.jsonl and SOURCES.md. Returns the summary numbers."""
    fetched = datetime.now(timezone.utc).date().isoformat()
    with requests.Session() as session:
        session.headers["User-Agent"] = USER_AGENT
        wiki_pool = collect_wiki_pool(session)
        speech_pool = collect_speech_pool(session)

    prompts, rest = choose_prompts(interleave(wiki_pool, speech_pool))
    text_docs = choose_text(rest)
    if len(prompts) < N_PROMPTS:
        raise RuntimeError(f"only {len(prompts)} usable prompts found, need {N_PROMPTS}; raise POOL_SIZE")
    if sum(c.word_count for c in text_docs) < TEXT_WORD_TARGET:
        raise RuntimeError("not enough text for the word target; raise POOL_SIZE")

    write_jsonl(out_dir / "sv_text.jsonl", [text_record(c) for c in text_docs])
    write_jsonl(out_dir / "sv_prompts.jsonl", [prompt_record(c, opening) for c, opening in prompts])
    checksums = {name: sha256_file(out_dir / name) for name in ("sv_text.jsonl", "sv_prompts.jsonl")}
    (out_dir / "SOURCES.md").write_text(render_sources_md(fetched, text_docs, prompts, checksums), encoding="utf-8")

    return {
        "fetched": fetched,
        "n_text_docs": len(text_docs),
        "n_text_words": sum(c.word_count for c in text_docs),
        "n_prompts": len(prompts),
        "checksums": checksums,
    }


if __name__ == "__main__":
    print(build_damage_data())
