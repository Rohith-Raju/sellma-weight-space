<img src="assets/icon.svg" width="64" height="64" alt="">

# Can a targeted weight edit improve a model's Swedish? A test on Gemma 3 4B

## Start here

| What | Where |
|---|---|
| The research note (short) | [`RESEARCH_NOTE.md`](RESEARCH_NOTE.md) |
| The full report | [`report/report.pdf`](report/report.pdf) (source: `report/report_full.md`) |
| How AI was used | `RESEARCH_NOTE.md` section 8 and the report's section 8, "AI usage" |
| The saved results, and which file backs which claim | [`results/`](results/README.md) |
| How to rerun and re-check everything | [`REPRODUCE.md`](REPRODUCE.md) (settings, versions and seeds: [`CONFIG.md`](CONFIG.md)) |
| Licence notices for the weight-change files and the datasets | [`NOTICE`](NOTICE) |
| The independent re-checks of every number | [`verification/`](verification/) (`SUMMARY.md` first) |

**The Högskoleprovet question texts are not in this repository** (they contain copyrighted third-party
texts; see [`report/DATA_RIGHTS.md`](report/DATA_RIGHTS.md)). The question ids, the answer letters and
every model answer are here. Rebuild the texts with `make data`: it downloads the free PDFs from
studera.nu (`bench fetch`), extracts the questions (`bench extract`) and checks the result against the
checksum in `exams/split.json` (`make check-data`).

## What the bench is

This bench measures how good a language model is at Swedish. It tests language skill, not knowledge of facts. It uses real past papers of Högskoleprovet, the Swedish university entrance exam. We run it on a model before and after we edit the model's weights, so we can see whether the edit helped. It also checks that the edit did not quietly damage the model.

Everything is plain files: the questions (after `make data`), the settings, the answers and the reports. You can open all of them in VS Code.

## The test

We use the verbal half of Högskoleprovet. It has three sections we use (the fourth, ELF, is English and is skipped):

| Section | Name in files | What it asks | What it measures |
|---|---|---|---|
| ORD (word meaning) | `ORD` | Pick the word closest in meaning to a given word. | Mostly vocabulary, which is closer to knowledge than to skill. |
| LÄS (reading) | `LAS` | Read a text, answer a question about it. | Understanding what a Swedish text says. |
| MEK (sentence completion) | `MEK` | Pick the words that fill the gaps in a sentence. | Grammar, word choice and how sentences hang together. |

Every question is multiple choice, with 4 or 5 options labelled A to E. Because ORD is mostly vocabulary, every report shows it on its own row, so you can read LÄS and MEK without it.

**Where the questions come from.** `bench fetch` downloads the booklets and the official answer keys from studera.nu, the official site for the exam. `bench extract` turns the PDFs into one record per question. The correct answer always comes from the official key. Nothing is typed in or fixed by hand.

**How many.** From `exams/extract_report.md`: 28 exams (2013 to 2026), 57 booklet sections ("provpass") found, 55 kept, **1650 questions**.

**What was set aside, and why.** Both provpass 5 booklets of 2018-04-14. That exam had two versions of the same provpass, and nothing printed on the booklets says which answer key belongs to which. Rather than guess, the whole provpass was left out. In general, if any check on a provpass fails, the whole provpass is set aside and the reason is written to the report.

## The split: which exams are used for what

A model may have seen an old exam during training. Then a good score could just mean it remembers the answers. So the exams are split by date:

| Split | Exams | Questions | Used for |
|---|---|---|---|
| `test` | 2025-10-19, 2026-04-18, 2026-10-18 | 180 (60 per section) | All results we report. Written after both models' training ended. |
| `gemma_extra` | 2024-10-20, 2025-04-05 | 120 | An extra check for Gemma only. Never used to choose an edit. |
| `dev` | the other 23 exams | 1350 | Trying out edits. |

Training cutoffs (the date a model's training data ends):

- Gemma 3: August 2024, as published by Google.
- Qwen3.5-9B: April 2025. This date was given by the project owner. Qwen has not published it. Because 2025-04-05 falls in that month, it counts as seen by Qwen.

The dates are set in `config/split.toml`. `bench split` writes the result to `exams/split.json` once, together with a checksum of the questions file. After that it refuses to change it (unless forced with `--force`, which we avoid). This stops the test set from drifting.

## How a question is scored

1. The question is written out in a fixed format: short Swedish instructions in the exam's own style, the text or sentence, the options as `A. ...` lines, then "answer with the letter only". The model's own chat format is applied on top (`src/bench/prompts.py`).
2. We do not let the model write an answer. We read the probability it gives to each option letter as its very next word (token). A token is the small piece of text a model reads and writes, often a word or part of a word.
3. The letter with the highest probability is the model's answer.

There is no random sampling, so the same model always gives exactly the same result.

Each answer records three numbers:

| Number | Meaning |
|---|---|
| accuracy | Share of questions where the top letter is the correct one. |
| prob_correct | The probability on the correct letter, counted among the option letters only. 1.0 = fully sure and right; 0.25 = no idea with 4 options. More sensitive than accuracy, because it also moves when the model becomes more or less sure. This is the number the decision rule uses. |
| prob_on_letters | How much of the model's probability went to option letters at all. Low (below 0.5, shown as "unsure") means the model wanted to say something else, so its letter ranking means less. |

## Damage checks

An earlier edit kept a 95.3% reading score while the model wrote looping nonsense and its perplexity exploded. The exam score alone missed it. So every run also does two checks:

- **Perplexity**: how surprised the model is by real Swedish text. Lower is better. A model that has lost its grip on the language scores much higher. The text is 75 Riksdag (Swedish parliament) speeches from the 2025/26 session, 40,154 words in all. They were given between 10 September 2025 and 12 August 2026, so neither model has seen them. The builder can also take Swedish Wikipedia articles, but the built files contain none ([`data/damage/SOURCES.md`](data/damage/SOURCES.md)).
- **Loop rate**: the model continues 50 short Swedish openings, up to 200 tokens each. A reply "loops" if some run of 4 words appears 3 or more times. The loop rate is the share of replies that loop.

Perplexity is only comparable within one backend (full precision vs full precision, or compressed vs compressed), because each backend computes it its own way.

The texts live in `data/damage/` and are built once with `uv run python -m bench.damage_data`. `data/damage/SOURCES.md` records sources, licences, Riksdagen's terms of use and checksums. Källa: Sveriges riksdag (Source: The Swedish Parliament).

## When does an edit count as an improvement?

The rules are in `config/decision.toml`. They were fixed before any edit was run. `bench compare` checks them, and all of them must pass:

1. **The score goes up for real.** On the test split, the average prob_correct rises, and the 95% confidence interval of the rise is entirely above zero. (A confidence interval is the range the true change very likely lies in. It is computed by a "paired bootstrap": resampling the same questions many times for both runs.)
2. **It beats a random edit of the same size.** Same test against a control run. With no control run, this rule fails.
3. **No damage.** Perplexity rises by at most 5%, and the loop rate rises by at most 2 percentage points.

A rule that cannot be checked counts as failed, and the report says why. Only the `test` split gives a verdict that counts; on other splits it is marked "indicative". Note that rule 1 uses all questions together, ORD included; the per-section rows are for information.

## How it fits together

```
 studera.nu
     |
   fetch  ──>  exams/raw/              PDFs + manifest.json
     |
  extract ──>  exams/questions.jsonl   one question per line (not in git; rebuilt by make data)
     |         exams/extract_report.md
     |
   split  ──>  exams/split.json        frozen: test / gemma_extra / dev
     |
    run   ──>  runs/<id>/answers.jsonl
     |           backends (config/models.toml):
     |             hf        full precision (Hugging Face transformers)
     |             llamacpp  compressed GGUF (llama.cpp)
     |             random / oracle / broken   dry stand-ins, no model
     |
   damage ──>  runs/<id>/damage.json   perplexity + loop rate (part of run)
     |
   report ──>  runs/<id>/summary.md
     |
  compare ──>  runs/comparisons/<edited>__vs__<baseline>/comparison.md
```

| File | What it does |
|---|---|
| `src/bench/cli.py` | The `bench` command and its sub-commands. |
| `src/bench/paths.py` | Where every file and folder lives. |
| `src/bench/records.py` | The two record types (Question, Answer) and JSONL read/write. |
| `src/bench/fetch.py` | Downloads verbal booklets and answer keys from studera.nu; keeps a manifest with checksums. |
| `src/bench/extract.py` | Reads the PDFs into questions, matches answer keys, sets aside bad provpass, writes the report. |
| `src/bench/split.py` | Freezes the split into `exams/split.json`. |
| `src/bench/prompts.py` | Writes the fixed question text and applies the model's chat format. |
| `src/bench/models.py` | Reads `config/models.toml`. |
| `src/bench/run.py` | Runs one model on one split, scores each question, writes the run folder. |
| `src/bench/backends/base.py` | The interface every backend implements (letter probabilities, perplexity, generate, describe). |
| `src/bench/backends/__init__.py` | Picks the backend for a model; caches model-file checksums. |
| `src/bench/backends/hf.py` | Full-precision model on the GPU via Hugging Face transformers. |
| `src/bench/backends/llamacpp.py` | Compressed GGUF model via llama.cpp's `llama-server` and `llama-perplexity`. |
| `src/bench/backends/dry.py` | Stand-ins with no model: random, oracle, broken. |
| `src/bench/damage.py` | Perplexity and loop-rate checks; writes `damage.json` and `generations.jsonl`. |
| `src/bench/damage_data.py` | Builds the held-out Swedish texts in `data/damage/`. |
| `src/bench/stats.py` | Accuracy, Wilson interval (for one run's accuracy), paired bootstrap (for changes). |
| `src/bench/report.py` | Writes run summaries and comparisons; applies the decision rule. |
| `config/split.toml` | Which exam dates are test and gemma_extra. |
| `config/models.toml` | Every model version the bench can run, with its backend and paths. |
| `config/decision.toml` | The decision rule. |
| `tests/` | Automated checks of every step. |

## Folder layout

```
exams/
  raw/<date>/            downloaded PDFs (not in git); raw/manifest.json (in git) has url, sha256, download time
  questions.jsonl        every question, one JSON record per line (not in git: holds the question
                         texts; rebuild with `make data`, checked against the checksum in split.json)
  extract_report.md      counts, and what was set aside and why
  split.json             the frozen split (large: it lists every question id)
data/damage/             sv_text.jsonl, sv_prompts.jsonl, SOURCES.md (held-out Swedish text)
models/                  model weights (on the GPU machine only, not synced)
runs/
  <time>_<model>_<split>/      e.g. 20261119T120000Z_qwen-9b-full_test/
    run.json             what was run: model settings, checksums, git commit, command, times
    answers.jsonl        one answer per question, with the letter probabilities
    damage.json          perplexity and loop rate
    generations.jsonl    the 50 free continuations, each marked looping or not
    summary.json         the numbers, per section and per exam date
    summary.md           the same, readable
  comparisons/<edited>__vs__<baseline>/
    comparison.json      all numbers and the verdict
    comparison.md        the same, readable
```

Tips for VS Code: `.jsonl` files have one record per line. Open a `.md` file and press Cmd+Shift+V for a readable preview. In file names, `LAS` means LÄS.

## How to use it

**Connect to the GPU machine.** In a terminal: `ssh sellma-gpu`. Or in VS Code: Cmd+Shift+P → "Remote-SSH: Connect to Host" → `sellma-gpu` → open the folder `/root/sellma`.

**Run long jobs inside tmux**, so they keep going if the connection drops:

```sh
tmux new -s bench          # start a session
# ... run commands ...
# Ctrl-b then d            # detach; the job keeps running
tmux attach -t bench       # come back later
```

**The pipeline** (from `/root/sellma`). The split is already frozen in `exams/split.json`; do not run `bench split` again. In a fresh clone, run `bench fetch` and `bench extract` (or `make data`) first, because the PDFs and `questions.jsonl` are not in git.

```sh
uv run bench fetch                      # download exams (or: --only 2026-10-18)
uv run bench extract                    # build questions.jsonl and extract_report.md
uv run bench split                      # freeze split.json (refuses if it exists)
uv run python -m bench.damage_data      # build data/damage/ once (needed before any run with damage checks)

uv run bench run qwen-9b-full --split test             # model names come from config/models.toml
uv run bench run gemma-4b-full --split dev --limit 20  # quick check on 20 questions
uv run bench run qwen-9b-full --split dev --skip-damage   # exam only, no damage checks

uv run bench compare runs/<edited-run> runs/<baseline-run> --control runs/<random-edit-run>
```

`bench run` prints the run folder when it finishes. `bench compare` prints the report and saves it under `runs/comparisons/`.

**Dry runs** check the plumbing without a GPU or a model:

| Model | Expected |
|---|---|
| `dry-random` | Accuracy near chance (about 20–25%). |
| `dry-oracle` | Accuracy 100%. Anything less means the scoring code is wrong. |
| `dry-broken` | Perfect answers, but looping text and huge perplexity. Compared against `dry-oracle`, it must fail the damage rules. |

```sh
uv run bench run dry-oracle --split dev --limit 30
uv run bench run dry-broken --split dev --limit 30
uv run bench compare runs/<..._dry-broken_dev> runs/<..._dry-oracle_dev>   # expect FAIL on 3a and 3b
```

**Run the tests:**

```sh
uv run pytest
```

## The planned edit cycle

1. Edit the model at full precision.
2. Test it (`hf` backend) against the unedited full-precision model.
3. Compress both the edited and the unedited model with llama.cpp, using the same settings for both (for example Q4_K_M). Add them to `config/models.toml`.
4. Test the compressed edited model against our own compressed unedited model, not against a compressed file downloaded from elsewhere.
5. Every time, also make a random edit of the same size and pass it as `--control`. This shows whether the gain comes from the edit itself or from any change at all.

## Known limits

- **Small test set.** 180 test questions can only detect fairly large changes. Small real gains may not pass rule 1.
- **Qwen's cutoff is not official.** April 2025 is the owner's figure, not Qwen's. If it is later, some test exams could have been seen.
- **Extraction uses rules of thumb.** Dropping author names under texts and joining words split across lines were checked, but they are heuristics and could be wrong in rare cases.
- **Old exams may have been memorised.** Past papers are public, so the models may have trained on the dev exams. To check: compare a model's dev score with its test score. A much higher dev score suggests memorisation.
