# Results (curated copy)

This folder is a small copy of the output folder of the GPU machine (the project there lived at
`/root/sellma`; the full copy on the owner's Mac is `gpu_results/`, which is not in git). The folder
layout is the same, so a path in a log, a `run.json` or a script that says `runs/...`,
`experiments/...`, `editor_runs/...`, `archive/...` or `logs/...` is found here under `results/...`.

Size in git: 48,063,188 bytes (about 48 MB), 1,283 files. Nothing in here was edited after the GPU
runs; files were only copied or left out (see "What was left out" below). **No Högskoleprovet question
texts are in git** (owner decision, `report/DATA_RIGHTS.md`): 17 files that repeat them are git-ignored
(item 9 below). The question ids, answer letters and every model answer are here; the texts are
rebuilt with `make data` (`REPRODUCE.md`, section 3).

Read-only analysis scripts reproduce their saved reports from this folder exactly:

```sh
SELMA_RESULTS=results uv run python analysis/proof_check.py      # = experiments/analysis/reports/proof_check.txt
```

and `make verify` re-runs the independent verification scripts on it (see `REPRODUCE.md`; most of them
need `make data` first). `make figures` redraws the report figures from `experiments/analysis/`.

The weight-change files (`*.safetensors`) are modifications of Gemma 3 4B it or Qwen3.5-9B. Their
licence notices are in `../NOTICE`.

## Layout

| Folder | What it holds |
|---|---|
| `runs/` | The 10 standard runs of the unedited models (`qwen-9b-full`, `qwen-9b-q4` on dev and test; `gemma-4b-full`, `gemma-4b-q4` on dev, test and gemma_extra), the 3 dry runs (`dry-random`, `dry-oracle`, `dry-broken`) and their 2 comparisons. Each run folder: `run.json` (model, checksums of weights and data, versions, command, times), `answers.jsonl` (one line per question: chosen letter, letter probabilities), `damage.json`, `generations.jsonl`, `summary.json`, `summary.md`. |
| `experiments/analysis/` | Rotation tests and alternatives: `rotation_proof.json` / `.out` (dev, unedited ×3, editor 3's change, random control), `rotation_first_run_dev.jsonl`, `heldout_position.json` / `.jsonl` (hidden + test papers), `other_exams/` (Belebele sv/en, MMLU maths), `rotation_choices.json` (rotation test that keeps which option text was chosen; unedited and editor 3, dev and test), `q4_rotation/` (rotation test at 4-bit on the hidden and test papers: unedited, editor 3, random control, 4 shards each), `reports/` (printed output of every read-only analysis script, including `choice_patterns.txt`, `edit_sizes.txt`, `q4_rotation_report.txt`, `q4_weight_check.txt`). |
| `experiments/group/gemma-4b-full/` | The group-trained edit: `split.json` (drill / validation / hidden papers), `trained/` (delta, training log, `train_result.json`), `control/` (random delta, seed 1), `baseline/`, run folders per split and precision, `comparisons/`, `results.md` / `results.json`. |
| `experiments/editor/trial_*/` | The four trial edits written by the AI editors, scored on drill, validation, hidden, test and gemma_extra at full precision and 4-bit, each with a random control of the same size: `edit.json`, `edited/`, `control/`, `comparisons/`, `results.md` / `results.json`. |
| `editor_runs/` | The three AI-editor sessions (`gemma_b10`, `gemma_b25`, `gemma_b50` = budgets of 10, 25, 50 passes) and the four trial edits they registered (`trial_*`). Per session: `run.json`, `ledger.jsonl` (every pass, its cost and folder), `final_edit.*` (the submitted change; all three are all-zero), `parameters.json`, `letter_token_ids.json`, `scripts/` (the editor's own code), `notes/` (the editor's notes, files up to 1 MB), `passes/NNN_*/` (`result.json`, `worker.log`; the `request.json` files hold the full exam prompts and are git-ignored, item 9). **Editor 3's change** (the edit tested in the main experiment) is `editor_runs/trial_b50_k/final_edit.safetensors` (+ `final_edit.json`, `parameters.json`). |
| `archive/single_question_edits/` | The first, stopped experiment: one rank-1 edit per wrong dev question (`gemma-4b-full/`, `qwen-9b-full/`), each with its random control (`edits/<question>/`), `baseline/`, Gemma's `combined/` edit, `dev_subset.json`, `logs/`. (`failure_cases.md` / `.jsonl` hold question text and are git-ignored, item 9.) |
| `logs/` | Terminal logs of the GPU machine (standard runs, compression, group training, other-exam runs, backend checks). `logs/rotation_proof.json`, `rotation_proof.out` and `rotation_test.jsonl` are where the scripts wrote them; they are byte-identical to the copies in `experiments/analysis/` (`rotation_test.jsonl` = `rotation_first_run_dev.jsonl`). |

## Which files back which claim

The claim groups are those of `verification/CLAIMS.md`; the independent checks are in
`verification/report_G1.md` … `report_G8.md` and `verification/SUMMARY.md` (which lists the corrections).

| Group | Claims | Files |
|---|---|---|
| **G1** Data, split, standard runs | 1,650 questions from 55 passes; split test / gemma_extra / dev; accuracy, perplexity, loop rate of the unedited models; right weights used | `runs/*/summary.json`, `answers.jsonl`, `damage.json`, `generations.jsonl`; weight and data checksums in `runs/*/run.json` (`describe.model_files`, `describe.sha256`, `questions_sha256`, `split_sha256`); data: `../exams/extract_report.md`, `../exams/split.json`; compression: `logs/compress_baselines.log`, `logs/recompress.log` |
| **G2** Option-A habit (original order) | Gemma picks A less often than it is the key; full vs 4-bit; 124 vs 9 sign test | `runs/20261119T060704Z_gemma-4b-full_dev/answers.jsonl`, `runs/20261119T061105Z_gemma-4b-q4_dev/answers.jsonl`; `experiments/analysis/rotation_proof.json`; printed: `experiments/analysis/reports/letters.txt`, `habit_q4.txt`, `proof_check.txt` |
| **G3** Single-question rank-1 edits | Gemma 5 of 10 fixed, Qwen 7 of 7; test headline per edit and per control; letter shift; perplexity | `archive/single_question_edits/{gemma-4b-full,qwen-9b-full}/edits/*/` (`search.json`, `targeted_*`, `control_*` run folders and `.json`/`.safetensors`), `baseline/`, `logs/experiment_*.log`; the list of failure cases (`failure_cases.md` / `.jsonl`) is only in the full copy, so `verification/scripts/G3_claims.py` is skipped by `make verify` on a clone |
| **G4** Group edit | Split sizes; training stopped at step 300, step 50 chosen; hidden fixes 6/117 vs control 1/117; test 70.0 → 70.8; 4-bit hidden net −2 | `experiments/group/gemma-4b-full/split.json`, `trained/training_log.jsonl`, `trained/train_result.json`, `results.md`, `results.json`, `comparisons/*/`, run folders under `baseline/`, `trained/`, `control/`; printed: `experiments/analysis/reports/group_cmp.txt`; `logs/group_gemma-4b-full.log` |
| **G5** AI editors | Passes used 5/10, 7/25, 11/50; all-zero final edits; no rule breaks; practice results of trial edits; editors' diagnoses; size of each trial edit | `editor_runs/gemma_b*/ledger.jsonl`, `final_edit.json`, `final_edit.safetensors`, `scripts/`, `notes/`, `passes/*/result.json`, `worker.log` (`passes/*/request.json` only in the full copy); trial sizes: `editor_runs/trial_*/final_edit.json`; printed: `experiments/analysis/reports/pass_cmp.txt` (see its caveat in `analysis/README.md`) |
| **G6** Trial edits, full precision and 4-bit | Hidden LAS+MEK accuracy, A-right, B/C/D-right, A picked, per trial and control; damage flags; 4-bit net for trial_b50_k | `experiments/editor/trial_*/results.md`, `results.json`, `edited/{hidden,test,q4_hidden,q4_test,...}/answers.jsonl`, `control/*/answers.jsonl`, `comparisons/`; unedited baseline: `experiments/group/gemma-4b-full/baseline/`; printed: `experiments/analysis/reports/edited_habit.txt` |
| **G7** Rotation tests on Högskoleprovet (main experiment) | Dev: three unedited repeats identical; by-position accuracy unedited / editor 3 / random; right-at-A gained 83 lost 0; held-out hidden and test: original, prompt note, option averaging, editor 3; held-out gains 27/0 and 11/0 | `experiments/analysis/rotation_proof.json`, `rotation_proof.out`, `rotation_first_run_dev.jsonl`, `heldout_position.json`, `heldout_position.jsonl`; edit: `editor_runs/trial_b50_k/final_edit.safetensors`; control: `experiments/editor/trial_b50_k/control/control_edit.safetensors` (the analysis scripts rebuild the same control with `random_control`, seed 1); printed: `experiments/analysis/reports/proof_check.txt`, `a_habit_all_tests.txt` |
| **G8** Other exams | By-position accuracy on Belebele sv / en and MMLU maths, unedited / editor 3 / random; gains and losses; letter picks; dataset revisions | `experiments/analysis/other_exams/{unedited,editor3,random}.json` (written by `analysis/other_exams_v1_first_run.py`), `{unedited,editor3}_with_choices.json` (written by `analysis/other_exams.py`, adds the chosen letter); revisions are inside each file (`revisions`); printed: `experiments/analysis/reports/oe_stats.txt`, `a_picks.txt`, `a_habit_all_tests.txt`; logs: `logs/other_exams_*.log` |

Later analyses (run after the main experiment; scripts and commands in `REPRODUCE.md`, section 9; the
numbers are re-checked in `verification/report_G9.md`):

| Output | Script | Backed by |
|---|---|---|
| `experiments/analysis/reports/knowledge_gap_vs_training.txt` (Gemma's never-right questions vs Qwen's answers) | `analysis/knowledge_gap_vs_training.py` | `experiments/analysis/rotation_proof.json`, `runs/*_qwen-9b-full_dev/answers.jsonl`, `runs/*_gemma-4b-full_dev/answers.jsonl` |
| `experiments/analysis/rotation_choices.json` | `analysis/rotation_choices.py` (GPU) | itself (per question and order: probability of each option text, chosen option, chosen position) |
| `experiments/analysis/reports/choice_patterns.txt` (knows / habit at A / unsure / wrong belief / position guess / no idea) | `analysis/choice_patterns.py` | `experiments/analysis/rotation_choices.json`, `experiments/group/gemma-4b-full/split.json` |
| `experiments/analysis/reports/edit_sizes.txt` (how many weights each edit changed) | `analysis/edit_sizes.py` | `archive/single_question_edits/*/edits/*/search.json`, `experiments/group/gemma-4b-full/trained/delta.json`, `editor_runs/trial_*/final_edit.json` |
| `experiments/analysis/reports/q4_weight_check.txt` (does 4-bit compression keep editor 3's change?) | `analysis/q4_weight_check.py` (GPU) | `editor_runs/trial_b50_k/final_edit.safetensors`, `experiments/editor/trial_b50_k/control/control_edit.safetensors`, the Gemma weights |
| `experiments/analysis/q4_rotation/*_shard*of4.jsonl` | `analysis/q4_rotation.py` (GPU) | itself |
| `experiments/analysis/reports/q4_rotation_report.txt` (4-bit rotation test next to full precision) | `analysis/q4_rotation_report.py` | `experiments/analysis/q4_rotation/`, `experiments/analysis/heldout_position.jsonl` |

## What was left out, and why

Everything below stays in the full copy (`gpu_results/` on the owner's Mac, or `/root/sellma` on the
GPU machine). Nothing that a number in the report is computed from was left out, except where noted.

1. **9 large activation files (≈ 15.3 GB)**, never copied off the GPU disk
   (`gpu_results/LARGE_FILES_LEFT_ON_GPU.txt`): the editors' read-pass outputs
   `editor_runs/gemma_b10/passes/00{1,2}_read/result.safetensors`,
   `editor_runs/gemma_b25/passes/00{1,2}_read/result.safetensors`,
   `editor_runs/gemma_b50/passes/00{1,2,3,4}_read/result.safetensors` (1.76–1.78 GB each) and
   `editor_runs/gemma_b50/notes/prefix_res.pt` (1.07 GB). They are activations the editors looked at;
   no reported number is computed from them.
2. **`prompts.jsonl` in every run folder (≈ 97 MB in total).** The exact prompt text of each question.
   It is derived: `src/bench/prompts.py` rebuilds it from `exams/questions.jsonl` and the model's chat
   template. It also repeats the question texts (see `report/DATA_RIGHTS.md`).
3. **Per-pass edit files `editor_runs/gemma_b*/passes/*_answer/edit.safetensors` (≈ 100 MB).** The
   editors' trial changes. The four that were registered for testing are kept as
   `editor_runs/trial_*/final_edit.*` (with the exceptions in 4). `verification/scripts/G5_verify.py`
   needs them and therefore runs only on the full copy.
4. **Two trial-edit tensors, for size:** `editor_runs/trial_b25_mlp/final_edit.safetensors` (15.7 MB)
   and `editor_runs/trial_b10_q_half/final_edit.safetensors` (5.2 MB). Their `final_edit.json` (size,
   explanation) are kept. `trial_b10_q_half` is exactly `trial_b10_q_full` with every value multiplied
   by 0.5 (same rows; checked element by element), so it can be rebuilt from the kept `trial_b10_q_full`.
5. **Random-control tensors of three trials** (`experiments/editor/trial_{b10_q_full,b10_q_half,b25_mlp}/control/control_edit.safetensors`,
   25.7 MB). `bench.editor.evaluate.random_control` (seed 1, see `CONFIG.md`) rebuilds them from the
   trial edit: for `trial_b50_k` the rebuilt control equals the saved one to within 8e-10 per value
   (float32 rounding; checked on a Mac CPU against the GPU machine's file). For the two trials whose
   tensors are left out (item 4) the edit must be rebuilt first. The control of editor 3's change
   (`trial_b50_k`, 2.6 MB) is kept.
6. **Editors' notes files over 1 MB** (`gemma_b10/notes/edit_v1.pt`, `edit_q_v1.pt`, `edit_q_v1_half.pt`;
   `gemma_b25/notes/neuron_pos_t.pt`, `neuron_t_mek.pt`; `gemma_b50/notes/nl0_selectivity.pt`; ≈ 30 MB). Intermediate
   tensors of the editors' own analysis. `verification/scripts/G5_notes_edits.py` needs `edit_q_v1.pt`.
7. **Repeated pass requests.** Where an editor sent the same request again, only the first copy is
   kept; the others are byte-identical (same sha256):

   | Left out | Identical to | sha256 |
   |---|---|---|
   | `editor_runs/gemma_b10/passes/004_answer/request.json` | `.../003_answer/request.json` | `1bc7af963d85…73f5` |
   | `editor_runs/gemma_b25/passes/00{4,5,6,7}_answer/request.json` | `.../003_answer/request.json` | `bb06bb13abcb…ff94` |
   | `editor_runs/gemma_b50/passes/00{6,7,8,9}_answer/request.json`, `011_answer/request.json` | `.../005_answer/request.json` | `c0e6b526b673…4ff8` |

8. **Not copied at all:** `gpu_results/analysis/` (same scripts as `analysis/` at the repository root,
   before the path change), `gpu_results/verification/` (same as `verification/scripts/R_*`), empty
   `lock` files and `__pycache__/`.
9. **Files that repeat the question texts (17 files, git-ignored).** They are on disk in the owner's
   copy but not in git (`.gitignore`, checked with `git check-ignore -v`):
   - the 13 kept pass requests `editor_runs/gemma_b{10,25,50}/passes/*/request.json` (about 17.6 MB;
     their `"texts"` field holds the full chat prompts with passage, question and options). Their other
     fields (`kind`, `question_ids`, `options_orders`, `layers`, `top_k`, `letters`) are lost with
     them. `analysis/pass_cmp.py`, `verification/scripts/G5_verify.py` and `G5_remote_readonly.py`
     read them, so they run only on the full copy. The saved output of `pass_cmp.py` is kept as
     `experiments/analysis/reports/pass_cmp.txt`;
   - `archive/single_question_edits/{gemma-4b-full,qwen-9b-full}/failure_cases.md` and `.jsonl`
     (the full prompts of the 10 failure cases per model). `verification/scripts/G3_claims.py` reads
     the `.jsonl` (only its `question_id` and `section` fields).
   Item 7 lists the byte-identical repeats of the pass requests; those were never copied.

## Notes for readers

- Headline = LÄS + MEK (`LAS`, `MEK` in files); ORD is reported separately.
- "Editor 3" = the `gemma_b50` session; its change is `trial_b50_k` (layer 23 `self_attn.k_proj`, rows).
- `pass_cmp.txt` scores passes in the original option order; for rotated passes use the editors' notes.
- Full-precision and 4-bit perplexities are computed differently and cannot be compared with each other.
- `git_commit` is `null` in every `run.json`: the code version of each run is not recorded.
