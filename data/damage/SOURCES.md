# Swedish text for the damage checks

Built by `bench.damage_data` on 2026-11-19 (fetch date). Rebuild with `uv run python -m bench.damage_data`.

## What was collected

* `sv_text.jsonl`: 75 documents, 40154 words, for perplexity.
* `sv_prompts.jsonl`: 50 prompts (the opening 30-60 words, 1-3 sentences) of other documents, not in `sv_text.jsonl`.

| Source | Documents (text) | Words (text) | Published | Licence |
|---|---|---|---|---|
| data.riksdagen.se | 75 | 40154 | 2025-09-10 to 2026-08-12 | Public parliamentary record, Riksdagen open data |

Every document was first published after 2025-05-01, after the training cutoffs of Gemma 3 (August 2024) and Qwen3.5 (April 2025).

## Sources and attribution

* **Swedish Wikipedia** (sv.wikipedia.org). Articles whose page creation is logged after the cutoff, text from the plain-text extract API. Licence CC BY-SA 4.0: credit the article title and URL (as listed per document in the JSONL files), and share adaptations under the same licence. The text is the article's current version when fetched, which may be newer than its creation.
* **Riksdagen open data** (data.riksdagen.se). Speeches (anföranden) in the 2025/26 session, read from the bulk file https://data.riksdagen.se/dataset/anforande/anforande-202526.json.zip (sha256 `6d3e0bb56427d28b3fe4f3b6c1ebccad9ba4679398219cd3012060630c23f28f`). Public parliamentary record. Cite the speaker, debate and riksdagen.se URL. Terms of use (Riksdagen's "Användningsvillkor", https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anvandarstod/anvandningsvillkor/, checked 2026-11-19): the open data may be used and redistributed freely as long as the source is given, as "Källa: Sveriges riksdag" or "Source: The Swedish Parliament". Do not suggest that the Riksdag is behind or endorses the work. The data contains personal data (the speakers' names and speeches), so the GDPR and lag (2018:218) apply to any further use.

Attribution for the Riksdagen text in this folder (and the prompts repeated in the `generations.jsonl` files of the run folders):

> Källa: Sveriges riksdag / Source: The Swedish Parliament

## Selection and cleaning

* Candidates per source are sorted by SHA-256 of their id (a fixed, date-independent order), then the two sources are interleaved one by one.
* The first 50 documents with a usable opening (30-60 words, 1-3 sentences) become prompts. They are excluded from the text.
* The rest, in the same order, fill the text until about 40000 words (at most 150 documents). Each text document has at least 150 words.
* Cleaning keeps prose paragraphs of at least 8 words that end with sentence punctuation. Drops headings, lists, tables, reference sections (everything after Referenser, Källor, Noter, Se även and similar), and stage directions such as (Applåder).

## Checksums

* `sv_prompts.jsonl`: `81b231ffb94673e9b05422cc111c32f58881d06326f1f3abfbb137ab5695277a`
* `sv_text.jsonl`: `6f13e43de548fd58397c600b38b417e92071fcdd01b9e8ae337f11d7c137e912`

## Caveats

* Upstream text can be edited after the fetch, so a rebuild on another date can differ. Compare the checksums above to see whether the files are the same.
* The sentence splitter is simple; a few prompts may end at an abbreviation or start mid-thought.
