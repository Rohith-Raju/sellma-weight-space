# Reproducing the results

Every command below is also a `make` target (`make help`). Configuration, versions, checksums and
seeds are in `CONFIG.md`; the saved results and which claim each file backs are in `results/README.md`.

| Step | Needs a GPU? | Make target |
|---|---|---|
| 1. Check the saved results (no models) | no | `make test`, `make reports`, `make figures`; after step 3 also `make check-data`, `make verify` |
| 2. Environment | no | `make env` |
| 3. Data (exam questions) | no | `make data` |
| 4. Models | no (download) | `make models`, `make check-models`, `make compress` |
| 5. **Main experiment**: rotation test, held-out alternatives, other exams | yes | `make main` |
| 6. Standard runs of the unedited models | yes | `make standard-runs` |
| 7. The edit experiments (group edit, editor trials, single-question edits) | yes | see below |
| 8. Re-verify the claims | no | `make verify`, `make verify-pdf` |
| 9. Later analyses (choice patterns, edit sizes, 4-bit rotation test) | some | see below |
| 10. Figures and the report PDF | no | `make figures`, `make report` |

**The Högskoleprovet question texts are not in this repository** (owner decision, option (b) in
`report/DATA_RIGHTS.md`). `exams/questions.jsonl` and the PDFs in `exams/raw/` are git-ignored. A fresh
clone must run `make data` (step 3, needs network access to studera.nu) before anything that reads the
questions: `make check-data`, most of `make verify`, `make main`, `make standard-runs` and the scripts
that run a model.

The original runs used one NVIDIA RTX PRO 6000 Blackwell Workstation Edition, torch 2.15.0+cu132
and transformers 5.23.0, models in bfloat16 (recorded in every `run.json`). The main experiment needs
only Gemma 3 4B it at full precision; the standard runs also need Qwen3.5-9B and the two 4-bit files.

## 1. Check the saved results without a GPU

```sh
uv sync                    # base environment (enough for everything in this section)
make test                  # = uv run pytest; unit tests of every bench step, no model needed
make reports               # re-run the 8 read-only analysis scripts on results/, diff against the saved reports
make figures               # redraw the 9 report figures from results/ and check the numbers against CLAIMS.md
make data                  # step 3: rebuild exams/questions.jsonl (not in git), then check-data
make check-data            # exams/questions.jsonl and exams/split.json match the checksums in the saved runs
make verify                # re-run the independent verification scripts (verification/scripts/) on results/
```

`make test`, `make reports` and `make figures` need no question texts. The real-PDF tests in
`tests/test_extract.py` skip when `exams/raw/` is missing.

`make reports` re-reads `results/` through `SELMA_RESULTS=results` and prints `same` for each report
that comes out byte-identical (all 8 do). Any single script can be run the same way:

```sh
SELMA_RESULTS=results uv run python analysis/proof_check.py
```

## 2. Environment

```sh
uv sync --extra gpu        # = make env; Python >= 3.11, versions pinned in uv.lock
```

**llama.cpp b8680** is needed only for the 4-bit (Q4_K_M) models: compressing (`make compress`) and
any run of `qwen-9b-q4` / `gemma-4b-q4`. The bench expects a CUDA build at `/root/llama.cpp`
(`src/bench/backends/llamacpp.py`, `src/bench/compress.py`) and the converter's own Python
environment at `/root/llama-venv`:

```sh
git clone https://github.com/ggml-org/llama.cpp /root/llama.cpp
git -C /root/llama.cpp checkout b8680            # commit 15f786e6581598638840276948a7e6183fc96a83
cmake -S /root/llama.cpp -B /root/llama.cpp/build -DGGML_CUDA=ON && cmake --build /root/llama.cpp/build -j --config Release
scripts/setup_llama_venv.sh                       # converter venv (Python 3.11) at /root/llama-venv
```

For `llama-server` / `llama-perplexity` elsewhere, set `llama_cpp_bin = "<dir with the binaries>"`
under the model in `config/models.toml`. The compression step's paths are fixed in
`src/bench/compress.py` (a symlink `/root/llama.cpp` is the simplest way to use another location).
The CUDA build flags of the original build were not recorded.

## 3. Data

```sh
uv run bench fetch         # booklets + answer keys from studera.nu -> exams/raw/ (manifest.json has url + sha256 per file)
uv run bench extract       # -> exams/questions.jsonl, exams/extract_report.md
make check-data            # the rebuilt questions.jsonl must match questions_sha256 in exams/split.json
```

(= `make data`.) **The question texts are not in this repository.** `exams/questions.jsonl` and the
PDFs in `exams/raw/` are git-ignored, because the booklets contain copyrighted third-party texts and
there is no reuse licence (owner decision 2026-11-19, option (b) in `report/DATA_RIGHTS.md`). What is
in git: the question ids and the frozen split (`exams/split.json`), the answer letters and every model
answer (`answers.jsonl` in `results/`), the extraction report (`exams/extract_report.md`) and the
download manifest (`exams/raw/manifest.json`: URL and sha256 per PDF). The three commands above
rebuild the texts from the free PDFs on studera.nu. `make check-data` then checks the rebuilt
`questions.jsonl` against `questions_sha256` in `exams/split.json` (`6d84c5ec…7955`). If they match,
the rebuild is byte-identical to the file the experiments used. If UHR has moved or changed a PDF,
`bench fetch` and the manifest checksums show which one.

`bench fetch` rewrites `exams/raw/manifest.json` with a new `fetched_at`, so git shows the file as
modified after `make data`. This is harmless: `make check-data` is the real check.

**Do not run `bench split`.** The split is frozen: `exams/split.json` was written
once (2026-11-19T00:55:55Z, from `config/split.toml`), stores the checksum of `questions.jsonl`, and
`bench split` refuses to overwrite it (`--force` exists but would change the test set).

The Swedish text for the damage checks (perplexity, loop rate) is in `data/damage/`, with sources
and checksums in `data/damage/SOURCES.md`. It was built once by `uv run python -m bench.damage_data`;
there is no need to rebuild it.

## 4. Models

```sh
uv run hf auth login       # Gemma is gated: accept its terms on Hugging Face first
make models                # google/gemma-3-4b-it @ 093f9f38... -> models/gemma-3-4b-it-official
                           # Qwen/Qwen3.5-9B      @ c2022362... -> models/Qwen3.5-9B-HF
make check-models          # every file against describe.model_files in results/runs/*/run.json
make compress              # our own Q4_K_M files -> models/compressed/{gemma-4b-full,qwen-9b-full}-Q4_K_M.gguf
```

The revisions come from a later check of the Hugging Face repositories, not from the download
itself (see `CONFIG.md`); the weight-file checksums recorded in every `run.json` are the binding
identity, and `make check-models` compares them. `make compress` runs
`convert_hf_to_gguf.py --outtype f32` then `llama-quantize ... Q4_K_M 32`, no importance matrix; the
result must have sha256 `fe8598adf4…` (Gemma) and `d4579f70b8…` (Qwen), also checked by
`make check-models`.

## 5. The main experiment

Question: is the effect of editor 3's weight change (`trial_b50_k`, a change to rows of layer 23's
`self_attn.k_proj` in Gemma 3 4B it) a removal of a position habit (overlooking the answer when it is
option A) rather than new Swedish ability, and how does it compare with inference-time alternatives?

```sh
make main
```

This runs, in order:

| Step | Command | Writes | Saved copy |
|---|---|---|---|
| prepare | `make prepare-main` | copies editor 3's change `results/editor_runs/trial_b50_k/` → `editor_runs/trial_b50_k/` (where `bench.editor.evaluate.edit_source` looks), the group split → `experiments/group/gemma-4b-full/split.json` (defines the 8 hidden papers), the first rotation run → `logs/rotation_test.jsonl` | |
| rotation test, dev | `uv run python analysis/rotation_proof.py` | `logs/rotation_proof.json`, `logs/rotation_proof.out` | `results/experiments/analysis/rotation_proof.json`, `.out` |
| held-out papers + alternatives | `uv run python analysis/heldout_position.py` | `experiments/analysis/heldout_position.json`, `.jsonl` | same paths under `results/` |
| other exams | `uv run --with datasets python analysis/other_exams.py unedited` (then `editor3`, `random`) | `experiments/analysis/other_exams/<condition>_with_choices.json` | `results/experiments/analysis/other_exams/` |
| compare | `uv run python scripts/check_main.py` | prints, per file, how many answers differ from the saved results | |

- **Rotation test** (`rotation_proof.py`): every 4-option dev question of LÄS and MEK (900) is shown
  4 times with the options rotated so the correct answer sits once at A, B, C and D. Conditions:
  unedited (twice, and once more after removing the change: determinism), editor 3's change, and a
  random change with the same tensors, rows and norm per tensor (`random_control`, seed 1).
- **Held-out** (`heldout_position.py`): the same test on the 8 hidden papers (320 LÄS+MEK) and the
  three test exams (120), plus two inference-time alternatives without any weight change: a prompt
  note ("the options are in random order; read all of them, also the first") and option averaging
  (each option text's probability summed over the 4 rotations).
- **Other exams** (`other_exams.py`): Belebele Swedish, the same 500 questions in English, and 400
  MMLU maths questions, at pinned dataset revisions (inside the script), unedited / editor 3 / random.

Expected output (from the saved results; `scripts/check_main.py` compares every answer):

| | A | B | C | D | right in all 4 |
|---|---|---|---|---|---|
| dev, unedited (3 identical repeats) | 53.6 | 75.9 | 72.0 | 70.8 | 405 of 900 |
| dev, editor 3's change | 62.8 | 73.4 | 70.1 | 69.9 | 439 |
| dev, random control | 53.8 | 75.8 | 72.0 | 70.9 | 406 |

Right-at-A with editor 3's change: 83 gained, 0 lost (dev); 27 / 0 on the hidden papers and 11 / 0 on
the test exams. On overall score, edit, option averaging and prompt note are statistically
indistinguishable on the held-out papers (`verification/SUMMARY.md`, point 4).

Scoring reads letter probabilities and never samples, so a rerun on the same GPU type and library
versions should match exactly (the saved repeats agree on all 3,600 answers). On other hardware a few
answers that sit within a hair of a tie may flip; `rotation_proof.out` counts 5 question-positions
within 1 point of a tie.

Outputs go under `SELMA_RESULTS` (default: the repository root; those folders are git-ignored), so a
rerun never overwrites `results/`.

## 6. The standard runs

```sh
make standard-runs         # or: scripts/phase1_baselines.sh (two chains in tmux, resumable)
```

The 10 runs, exactly as recorded in each `run.json` (`command`):

```sh
uv run bench run qwen-9b-full  --split dev
uv run bench run qwen-9b-full  --split test
uv run bench run qwen-9b-q4    --split dev
uv run bench run qwen-9b-q4    --split test
uv run bench run gemma-4b-full --split dev
uv run bench run gemma-4b-full --split test
uv run bench run gemma-4b-full --split gemma_extra
uv run bench run gemma-4b-q4   --split dev
uv run bench run gemma-4b-q4   --split test
uv run bench run gemma-4b-q4   --split gemma_extra
```

Each writes `runs/<time>_<model>_<split>/`; compare `summary.json` and `answers.jsonl` with the same
run in `results/runs/`. The dry runs that check the plumbing (`dry-random`, `dry-oracle`,
`dry-broken`) are described in `README.md`.

## 7. The edit experiments

These produced the inputs of the main experiment and the remaining claims (G3–G6). They write under
`experiments/` and `editor_runs/` at the root.

| Experiment | Command (as recorded in `run.json`) | Saved results |
|---|---|---|
| Group edit (rank-8 change trained on many wrong answers, seed 0; control seed 1) | `uv run bench group all --model gemma-4b-full` | `results/experiments/group/gemma-4b-full/` |
| Scoring an editor's trial edit (full precision, 4-bit, random control) | `uv run bench editor-eval --run trial_b50_k` (needs `editor_runs/trial_b50_k/`, see `make prepare-main`) | `results/experiments/editor/trial_b50_k/` |
| Single-question rank-1 edits (stopped early) | `uv run bench experiment all --model gemma-4b-full --combined` (and `qwen-9b-full`); see `scripts/overnight.sh` | `results/archive/single_question_edits/` (moved there after the run) |

The AI-editor sessions themselves (`results/editor_runs/gemma_b10`, `gemma_b25`, `gemma_b50`) were
done by AI agents working through the pass-limited workbench described in `docs/EDITOR_WORKBENCH.md`;
they cannot be replayed exactly. Their scripts, notes, ledgers and results are saved. Their pass
requests (`passes/*/request.json`) are not in git, because the `"texts"` field holds the full exam
prompts; they stay in the full copy `gpu_results/`.
Only `trial_b10_q_full` and `trial_b50_k` of the four trial edits ship as tensors (see
`results/README.md`, "What was left out").

## 8. Re-verifying the claims

`verification/` holds the independent re-check of every reported number (`CLAIMS.md`,
`report_G1.md` … `report_G8.md`, `SUMMARY.md`).

```sh
make verify                # 13 scripts that need only saved files (runs `make data` first if the exam questions are missing)
make verify-pdf            # 3 more that read the exam PDFs (answer keys, booklet text)
uv run --with datasets python verification/scripts/G8_datasets.py   # downloads Belebele + MMLU
```

Each verification script finds the repository from its own location (or from `SELMA_ROOT` if set) and
reads `gpu_results/` when that full private copy exists, otherwise `results/`. So the scripts run from any
folder and on a fresh clone, with no path editing. On `results/` they print exactly what they print on
the full copy. The exam questions are not in git: if `exams/questions.jsonl` is missing, `make verify`
and `make verify-pdf` first run `make data`, which downloads the public PDFs from studera.nu, extracts
the questions and checks them against the checksum in `exams/split.json`. Not runnable from `results/` alone:
`G5_verify.py` and `G5_notes_edits.py` (need the per-pass edit files and large notes left out for
size), `G5_remote_readonly.py` and `R_rescore.py` (run on the GPU machine with the model; the saved
output is `verification/scripts/R_rescore.log` and `R_results.json`), `G1_render_keys.py` (renders
PDFs to PNG for reading by eye: `uv run python verification/scripts/G1_render_keys.py OUTDIR exams/raw/<date>/<key>.pdf`).

**Skipped scripts.** Before each script, `make verify` and `make verify-pdf` check whether the script
reads a git-ignored input that is missing, and then skip it with a `SKIPPED <script>: needs ...` line
instead of stopping. A list of the skipped scripts is printed at the end. The inputs are:

- `exams/questions.jsonl` (8 of the 13 scripts, and `G1_booklet_text`, `G1_visual_keys`): run `make data`;
- the PDFs in `exams/raw/` (`G1_data_split`, `G1_answer_keys`, `G1_booklet_text`): run `make fetch`;
- `failure_cases.jsonl` in `results/archive/single_question_edits/<model>/` (`G3_claims`): **only in
  the full GPU copy**. It holds question text and is never published, so `G3_claims` is always skipped
  on a fresh clone. It uses only the `question_id` and `section` fields; its result is in
  `verification/report_G3.md`.

So on a fresh clone without `make data`, 5 scripts run (`G2_rotation_counts`, `G4_q4_ci_seeds`,
`G6_q4_diff`, `G7_ties_and_averaging`, `G8_recompute`). After `make data`, 12 of the 13 run.

`verification/` also holds later checks and reviews, not run by `make verify`:

| File | What it is |
|---|---|
| `report_G9.md` | Independent recomputation of the numbers in `RESEARCH_NOTE.md` and `report/report_full.md` from the per-question files in `results/` (scripts `verification/scripts/G9_kinds.py`, `G9_q4.py`, `G9_single.py`, `G9_older.py`) |

**Full GPU copy only.** These need files that are not in git, because they hold question text:
`analysis/pass_cmp.py` (reads `passes/*/request.json`; its saved output is
`results/experiments/analysis/reports/pass_cmp.txt`), `verification/scripts/G3_claims.py`
(`failure_cases.jsonl`), and `G5_verify.py` and `G5_remote_readonly.py` (`request.json`).

## 9. Later analyses

These were run after the main experiment. The ones that run a model need the GPU, Gemma 3 4B it in
`models/` and `make data`. The others read only saved files in `results/`.

| Script | Runs a model? | Command | Saved output |
|---|---|---|---|
| `analysis/rotation_choices.py` | yes (GPU) | `uv run python analysis/rotation_choices.py` | `results/experiments/analysis/rotation_choices.json` (the rotation test that also keeps which option text was chosen, unedited and editor 3, dev and test) |
| `analysis/choice_patterns.py` | no | `SELMA_RESULTS=results uv run python analysis/choice_patterns.py` | `results/experiments/analysis/reports/choice_patterns.txt` |
| `analysis/edit_sizes.py` | no | `SELMA_RESULTS=results uv run python analysis/edit_sizes.py` | `results/experiments/analysis/reports/edit_sizes.txt` |
| `analysis/q4_weight_check.py` | yes (GPU, llama.cpp b8680) | `uv run --with gguf python -I analysis/q4_weight_check.py` | `results/experiments/analysis/reports/q4_weight_check.txt` (does 4-bit compression keep editor 3's change?) |
| `analysis/q4_rotation.py` | yes (GPU, llama.cpp b8680; GGUFs from `q4_weight_check.py`) | `uv run python -I analysis/q4_rotation.py unedited\|edited\|control SHARD 4` | `results/experiments/analysis/q4_rotation/*_shard*of4.jsonl` |
| `analysis/q4_rotation_report.py` | no | `uv run python -I analysis/q4_rotation_report.py` | `results/experiments/analysis/reports/q4_rotation_report.txt` |
| `analysis/knowledge_gap_vs_training.py` | no | `uv run python analysis/knowledge_gap_vs_training.py` | `results/experiments/analysis/reports/knowledge_gap_vs_training.txt` |

`rotation_choices.py` writes under `SELMA_RESULTS` (default: the repository root). The last four
scripts use `gpu_results/` when it exists (the owner's Mac). Otherwise they read their inputs from the
repository root if they are there (the GPU machine), and else from `results/` (a fresh clone). They
always write under `gpu_results/` or the repository root, never into `results/`, so on a clone the
output lands in the git-ignored `experiments/analysis/` at the root, where it can be compared with the
saved copy. `choice_patterns.py` and `edit_sizes.py` only print their report; the saved `.txt` files hold
that printed output. These scripts are not part of `make reports`, because their saved output has not yet
been re-made through it.

## 10. Figures and the report PDF

```sh
make figures               # = uv run --with matplotlib python -I report/figures/make_figures.py
make report                # = bash report/latex/build.sh
```

`make figures` reads only saved files: `gpu_results/experiments/analysis/` if that folder exists,
else `results/experiments/analysis/` (the same files). Before drawing, it recomputes the key numbers
and compares them with `verification/CLAIMS.md`; it must print "All recomputed values match
verification/CLAIMS.md." It rewrites `report/figures/fig*.png` and `fig*.svg`. The PNGs come out
byte-identical; the SVGs differ only in their timestamp and internal ids.

`make report` builds `report/report.pdf` from `report/report_full.md` with pandoc and xelatex, using
`report/latex/defaults.yaml` and `header.tex`. It needs pandoc, xelatex and the fonts Charter,
Helvetica Neue and Menlo. It copies `report/figures/` into
`report/latex/figures/` and writes `report/latex/report_full.tex`, `.pdf`, `.log` and `build.log`;
these are git-ignored. The published copy is `report/report.pdf`.

## Change note: configurable paths in `analysis/*.py` (2026-11-19)

The analysis scripts were run on the GPU machine from `/root/sellma`, and that path was written into
them. To make them runnable from any checkout, the hard-coded `/root/sellma` was replaced with two
settings defined in `analysis/_paths.py`:

- `SELMA_ROOT`: the project folder (default: the repository root, found from the script's location);
- `SELMA_RESULTS`: the folder holding `runs/`, `experiments/`, `logs/`, `editor_runs/` (default:
  `SELMA_ROOT`; set `SELMA_RESULTS=results` to read the saved results).

Changed: `a_habit_all_tests.py`, `a_picks.py`, `edited_habit.py`, `group_cmp.py`, `habit_q4.py`,
`heldout_position.py`, `letters.py`, `oe_stats.py`, `other_exams.py`, `other_exams_v1_first_run.py`,
`pass_cmp.py`, `proof_check.py`, `rotation_proof.py`, `rotations.py`. In each, only the path prefix
changed (`/root/sellma/X` → `RESULTS / X`, or `ROOT / "src"` in `pass_cmp.py`); `rotation_proof.py`
and `rotations.py` also create `logs/` if it is missing, and the usage line of the two `other_exams`
scripts now names `analysis/` (they were first run from `work/`). Nothing they compute changed: the 8
read-only scripts reproduce their saved reports byte for byte from `results/` (`make reports`). The
scripts in `gpu_results/analysis/` (not in git) are the versions before this change.
`analysis/knowledge_gap_vs_training.py` was written later with its own rule (it reads `SELMA_ROOT`,
and `gpu_results/` inside it when that folder exists) and was not changed then.

## Change note: fresh-clone fallbacks (2026-11-19)

`report/figures/make_figures.py`, `analysis/knowledge_gap_vs_training.py`, `q4_rotation.py`,
`q4_rotation_report.py` and `q4_weight_check.py` used `gpu_results/` when it exists and otherwise the
repository root, so on a fresh clone they failed or read the wrong folder. Each now falls back to
`results/` for its inputs when they are not found there (see section 9). When `gpu_results/` exists,
or on the GPU machine where the inputs sit at the root, they read and write exactly as before.
`make verify` now skips scripts whose git-ignored input is missing (section 8) instead of stopping.
