# Data rights: Högskoleprovet questions in a public repo

Status: **DECIDED (2026-11-19): option (b).** The question texts are not published. See "Decision" at the end. This note sets out the facts and the options. It is not legal advice.
The assignment says: "Use only data and model weights that you have the right to use and share."
Checked 2026-11-19. Related notes are in `report/references/article_sources_licences.md`, in the entries [uhr_fpn], [uhr_booklet], [uhr_ht2025] and [sfs1960:729] and in the section "Högskoleprovet: is not republishing the right call?".

## 1. What UHR publishes, and what the booklets say

- Universitets- och högskolerådet (UHR) runs studera.nu. It offers past booklets ("provhäften") and answer keys ("facit") as free PDFs: "Det är gratis att ladda ner tidigare provhäften och facit." (https://www.studera.nu/hogskoleprov/om/forbereda/tidigare/)
- `bench fetch` downloaded 93 PDFs, 57 booklets and 36 answer keys, from 28 exam dates (2013-10-26 to 2026-10-18). They come from `https://www.studera.nu/globalassets/05-hogskoleprovet/hp-<date>/...`, for example `.../hp-2013-10-26/provpass1verbutanelf.pdf`. Every URL, with its sha256, is in `exams/raw/manifest.json`.
- **The notice in the booklets.** It appears on page 1 of 56 of the 57 booklets and is worded the same from 2013 to 2026:
  > "Tillstånd har inhämtats att publicera det upphovsrättsligt skyddade material som ingår i detta prov."
  > (English: "Permission has been obtained to publish the copyrighted material included in this test.")
  - Old layout: `exams/raw/2013-10-26/provpass1verbutanelf.pdf` p. 1, and the same on p. 1 of `provpass4verbutanelf.pdf`.
  - Middle years: `exams/raw/2019-04-06/provpass-1-verb-utan-elf.pdf` p. 1, and `exams/raw/2021-03-13/provpass-1-verb-utan-elf.pdf` p. 1.
  - Test split, new layout: `exams/raw/2026-04-18/provpass-2-verb-utan-elf.pdf` p. 1 and `exams/raw/2026-10-18/provpass-2-verb-utan-elf.pdf` p. 1. Both passes of each date have it.
  - The exception is `exams/raw/2019-10-20/provpass-5-verb-utan-elf.pdf`, which has no such sentence anywhere. Its 30 questions are in `questions.jsonl`.
- No booklet or answer key has a © line, a UHR name or address, or any terms for reuse or reproduction. The answer keys are bare letter tables.
- The notice says that **UHR** obtained permission to **publish**. It says nothing about anyone else reusing the material.
- On the HT 2025 FPN page (https://www.studera.nu/hogskoleprov/fpn/provfragor-facit-och-normering-hosten-2025/), UHR writes: "Universitets- och högskolerådet (UHR) har bara tillstånd att visa de engelska texterna (ELF) i en vecka efter provdagen." (English: "UHR only has permission to show the English texts (ELF) for one week after the test day.") This is why the archived files are "utan ELF". It also shows that UHR's licences for third-party texts are limited in time and scope.
- The same page links a source list, "Källor till HP 2025 B" (`.../hp-2025-10-19/kallhanv.pdf`). It names third-party sources for the LÄS passages and for individual MEK sentences. Examples: Birgitta Lindqvist, *Kinesisk ask* (Norstedts 1997); Johannes Edfelt, *Dikter* (Bonniers 2004); Svenska Dagbladet 2016-03-14; Sveriges Natur 2019-03-01.
- In `questions.jsonl` the LÄS passages keep their author bylines, for example "Frida Lundberg". The ORD items, single words with five options, are probably UHR's own work.
- I found no terms of use for exam material on studera.nu.
- UHR contact, from uhr.se: registrator@uhr.se, phone 010-470 03 00, Box 4030, 171 04 Solna. The booklets and the manifest contain no contact details.

## 2. Repo files that contain question text

Paths under `editor_runs/`, `archive/` etc. are inside `results/` (the curated copy of `gpu_results/`, same structure; see `results/README.md`).

Since the decision, every file in the table below except the last three rows is listed in `.gitignore` and is not in the public repository. Each one was checked with `git check-ignore -v`.

**Contain question text:**

| Path | Content |
|---|---|
| `exams/questions.jsonl` | All 1,650 questions (550 ORD, 550 LÄS, 550 MEK): stems, options and answer keys. Also 175 distinct LÄS passages, about 857,000 characters, with bylines. Size 5.0 MB. |
| `results/editor_runs/{gemma_b10,gemma_b25,gemma_b50}/passes/*/request.json` | 13 files in `results/` (byte-identical repeats left out), about 18.5 MB. The `"texts"` field holds 10,556 full chat-formatted prompts: instruction, passage, question and options. (The full copy `gpu_results/` has 23 such files, 15,676 prompts.) |
| `results/archive/single_question_edits/{gemma-4b-full,qwen-9b-full}/failure_cases.md` | The full prompts for the 10 failure cases in each file, including the whole LÄS passages ("The questions (Swedish, as the model saw them)"). |
| `results/archive/single_question_edits/{gemma-4b-full,qwen-9b-full}/failure_cases.jsonl` | The same 10 questions per model, in a `"question"` field. |
| `prompts.jsonl` (165 files) | Full prompts. These are **left out** of `results/` (they stay in `gpu_results/`, which is not in git). |
| `tests/test_extract.py` | Short real fragments of 2025-10-19 p3: one option sentence (LAS/11 A), two passage titles, and a shortened stem. Arguably a short quotation. **Still in git; open for the owner** (keep under 22 § or shorten). |
| `tests/test_prompts.py`, `tests/test_dry_backends.py` | Made-up example items plus the bench's own instruction wording. No exam text. |
| `src/bench/prompts.py` | The bench's own instruction wording only. |

**Checked and found to contain only ids, letters, probabilities or numbers:**
- `exams/split.json`: question ids and the sha256 of `questions.jsonl`.
- `exams/extract_report.md`: exam, pass, file name and counts.
- All 174 `answers.jsonl` files. Their keys are `question_id, section, correct_letter, chosen_letter, is_correct, letter_probs, prob_on_letters, prob_correct`.
- `experiments/analysis/*`.
- `editor_runs/*/notes/*`: the JSON, .md and log files, and grep over the .pt and .pkl files for prompt strings.
- All `*.safetensors` files.
- `verification/report_*.md`, `CLAIMS.md` and `SUMMARY.md`.
- `report/`, `docs/` and `README.md`.

The `generations.jsonl` files match only common Swedish words. They are model replies to Riksdagen speeches from the damage check, and `data/damage/SOURCES.md` covers that data separately.

Method: I grepped for the prompt phrases ("Svara endast med bokstaven", "Välj det alternativ"). I also matched every text file against substrings of the stems, options and passages in `questions.jsonl`. PNG figures were not checked, but they are plots made by `report/figures/make_figures.py`.

## 3. Legal considerations

This section is cautious and not legal advice.
- **Third-party texts.** Under upphovsrättslagen (1960:729), the LÄS passages and many MEK sentences are excerpts from works whose authors or publishers hold the rights (life plus 70 years). UHR's permission, as worded, covers UHR's own publication. Nothing suggests it extends to republishing by third parties. The one-week ELF licence points the other way.
- **UHR's own items and the collection.** ORD items and UHR-written stems and options may be protected works in themselves if they are original enough. A collection of 1,650 items may also fall under catalogue protection (49 §, a large compilation of information or the result of a substantial investment, 15 years). If 49 § applies, all 2013–2026 booklets are inside that period. Whether either protection applies is uncertain.
- **Allmän handling is not a reuse licence.** The offentlighetsprincipen gives a right to access and copies of public documents. My understanding is that it leaves copyright in those documents untouched. 26 a § removes protection only from laws and regulations, authorities' decisions, statements by Swedish authorities, and official translations of these. Exam booklets do not plausibly fit those categories, and works by outside authors inside them would not be freed anyway. This reading is **uncertain**.
- **Exceptions that do help.** 15 a and 15 b § (text and data mining) allow copying for our own evaluation, but copies may not be kept longer than needed. 22 § allows quotation "i överensstämmelse med god sed och i den omfattning som motiveras av ändamålet", which covers a few illustrative items. Neither exception covers redistributing the whole set.
- **Reproducibility without republishing.** The PDFs are free online. Anyone can run `bench fetch` and then `bench extract` to rebuild `questions.jsonl` deterministically, and check the result against `questions_sha256` in `exams/split.json`. The risk is that UHR moves or removes files. The manifest records URLs and sha256 values, so drift would be detected, but files lost from the site could not be recovered.

## 4. Options

**(a) Include everything, as now:** `questions.jsonl` plus the prompts in `request.json`.
- Pros: fully self-contained, and reviewers can inspect every item.
- Cons: publishes third-party copyrighted texts and the full item collection, with no licence we can point to. This is hard to square with the assignment's "right to use and share". `request.json` repeats the same texts about 10,500 times, adds about 18.5 MB, and would have to stay in sync with whatever is decided for `questions.jsonl`.

**(b) Ids, answer keys and a few quoted examples; everything else rebuilt locally.**
- Ship `split.json` (ids plus checksum) and `answers.jsonl` as they are.
- Optionally ship an answer-key-only file (id to letter). The answer keys are bare letter tables, so the risk there is lower but not zero (see 49 §).
- Quote 2 or 3 short items in the README under 22 §, with the source named.
- Readers rebuild with fetch plus extract and verify against the checksum.
- Pros: keeps the results reproducible and checkable without redistributing protected text.
- Cons: depends on studera.nu staying up, and needs an extra setup step.
- Changes needed:
  - add a line `/exams/questions.jsonl` to `.gitignore` (the commented line there cannot simply be uncommented, because git has no end-of-line comments);
  - remove `"texts"` from every `request.json`, or drop those files;
  - drop `failure_cases.jsonl`, or remove its `"question"` field;
  - cut `failure_cases.md` down to ids and short quotes, with no full passages;
  - optionally shorten the real fragments in `tests/test_extract.py`, or mark the test as requiring `exams/raw`.

**(c) Ask UHR for written permission** (registrator@uhr.se).
- Pros: the only route to clean redistribution of UHR's own items.
- Cons: UHR probably cannot license the third-party passages, so the answer may cover ORD only, or nothing. It is also unlikely to arrive before the deadline. Until it does, (b) applies.

**Model licences.** These are items to check, not decisions.
- `config/models.toml` records no licences. The references note says Gemma 3 4B IT is under the Gemma Terms of Use and Qwen3.5-9B is Apache-2.0.
- `results/` ships 32 weight-delta files (`*.safetensors`, about 15 MB in total). 18 are Gemma edits: `editor_runs/*/final_edit` (5 files; the three sessions' files are all-zero), `experiments/editor/trial_b50_k/control/control_edit`, `experiments/group/gemma-4b-full/{trained,control}/delta`, and the 10 files of the Gemma `single_question_edits`. 14 are Qwen edits (`single_question_edits`). The full copy `gpu_results/` (not in git) has more, including the per-pass edits.
- A Gemma delta may count as a "Model Derivative". If so, §3.1 of the terms requires us to:
  - pass on the use restrictions;
  - include a copy of the terms;
  - mark the modified files;
  - add the notice "Gemma is provided under and subject to the Gemma Terms of Use found at gemma.ai/resources/terms".
- The Qwen deltas need Apache-2.0 attribution, the licence text and a note of the changes.

## Decision (owner, 2026-11-19)

**Option (b).** The Högskoleprovet question texts are not published in this repository.

What is published:
- question ids and the frozen split (`exams/split.json`, which also stores `questions_sha256`, the sha256 of `questions.jsonl`);
- the answer letters and every model answer (`answers.jsonl` in each run folder: `correct_letter`, `chosen_letter`, letter probabilities);
- all run results and analyses in `results/`;
- the download manifest `exams/raw/manifest.json` (per file: URL, sha256, size, fetch time; no exam text);
- a rebuild path: `make data` runs `bench fetch` (the PDFs from studera.nu) and `bench extract` (rebuilds `exams/questions.jsonl`), then `make check-data` checks the rebuilt file against the checksum in `exams/split.json`.

Two short example items are quoted in the report under the quotation right (22 § upphovsrättslagen), with UHR named as the source.

What is not published (all in `.gitignore`): `exams/questions.jsonl`, the PDFs in `exams/raw/`, the editors' pass requests (`passes/*/request.json`), `failure_cases.md` and `failure_cases.jsonl`, and every `prompts.jsonl`. They stay in the full copy `gpu_results/` on the owner's Mac.

Checklist:
- [x] Choose (a), (b) or (c) for `exams/questions.jsonl`: (b). `.gitignore` has the line `/exams/questions.jsonl` and leaves out `exams/raw/*` except the manifest.
- [x] `request.json` files and `failure_cases.md` / `.jsonl`: left out whole (ignored), not stripped. Consequences: `analysis/pass_cmp.py`, `verification/scripts/G3_claims.py`, `G5_verify.py` and `G5_remote_readonly.py` need the full GPU copy (`REPRODUCE.md`, section 8).
- [ ] The fragments in `tests/test_extract.py`: still open (keep under 22 §, or shorten).
- [x] Quoted examples: two items in the report, under 22 §, source UHR. Fetch, extract and checksum check are documented in `REPRODUCE.md` (section 3) and `README.md`.
- [x] Gemma deltas: treated as Model Derivatives as a precaution. `NOTICE` at the repository root lists the modified files and carries the Gemma notice, the "modified files" statement, the Prohibited Use Policy link and the link to the terms. It also has the Apache-2.0 attribution for the Qwen deltas.
- (c) was not pursued. No request was sent to UHR.
