# Configuration, versions and seeds

Every value below was read from a file in this repository. The "Source" column names that file. Paths under `results/` are the curated copy of the GPU machine's output folder (on the GPU machine the project lived at `/root/sellma`). "Not recorded" means no file in the repo holds the value.

## 1. Models, data and tools

### Models

| Item | Identifier / revision | Checksum (sha256) | Licence | Source |
|---|---|---|---|---|
| Gemma 3 4B it (full precision) | HF `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767` (commit "Update README (#25)", 2025-03-21). Local folder `models/gemma-3-4b-it-official`, loaded as `Gemma3ForConditionalGeneration` | `model-00001-of-00002.safetensors` `eb5fd5e97ddd07b56778733e9653c07312529cb00980a318fc3e1c4e3b5a8f1f`; `model-00002-of-00002.safetensors` `fdde0e5aa5ced0fa203b3d50f4ab78168b7e3a3e08c6349f5cc9326666e1bb13`; `config.json` `9059f680…7ff`; `tokenizer.json` `4667f208…795`; `chat_template.json` `fe16baf7…cd9` | Gemma Terms of Use (HF licence tag "gemma", gated) | Revision and licence: `report/references/article_sources_licences.md` ([gemma3_hf], [gemma_terms]). Hashes: `describe.model_files` in `results/runs/*gemma-4b-full*/run.json` |
| Qwen3.5-9B (full precision) | HF `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` ("Update README.md", 2026-03-02). Local folder `models/Qwen3.5-9B-HF`, loaded as `Qwen3_5ForConditionalGeneration` | `model.safetensors-00001-of-00004.safetensors` `db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e`; `-00002-` `31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330`; `-00003-` `7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec`; `-00004-` `b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6`; `config.json` `d0883072…b05` | Apache-2.0 | Revision and licence: `report/references/article_sources_licences.md` ([qwen35_hf]); licence also in `report/references/methods_models_benchmarks.md`. Hashes: `results/runs/*qwen-9b-full*/run.json` |
| Gemma Q4_K_M GGUF (made by us) | `models/compressed/gemma-4b-full-Q4_K_M.gguf`, 2,489,894,176 bytes, created 2026-11-19T05:41:08Z | `fe8598adf4f20d50d88bf8877540013717051e869faff95117de56753a8cd127` | Derivative of Gemma: Gemma Terms of Use apply | `results/logs/recompress.log`; `describe.sha256` in `results/runs/*gemma-4b-q4*/run.json` |
| Qwen Q4_K_M GGUF (made by us) | `models/compressed/qwen-9b-full-Q4_K_M.gguf` (size not recorded) | `d4579f70b8f7418d95051b506d6940c9a14ac2b5dc4e0a9c276b6d8b10a9c2d3` | Derivative of Qwen3.5: Apache-2.0 | `results/logs/recompress.log`; `results/runs/*qwen-9b-q4*/run.json` |

How the GGUF files were made (`src/bench/compress.py`, `results/logs/recompress.log`): `convert_hf_to_gguf.py <hf folder> --outtype f32`, then `llama-quantize <f32 file> <out> Q4_K_M 32` (32 = threads). No importance matrix (`importance_matrix: null`). Same settings for both models. The first build (`results/logs/compress_baselines.log`) used `--outtype bf16` as the intermediate; the rebuild with `f32` gave byte-identical files (same sha256 for both, listed in `recompress.log`). A scratch build from a folder holding only symlinked weight files gave a different hash (`8af43ccd…dc0a`) and was not used. The converter ran in a separate Python 3.11 venv (`scripts/setup_llama_venv.sh`) with torch 2.6.0+cpu, transformers 4.57.6, numpy 1.26.4, sentencepiece 0.2.3, gguf 0.19.0 (`converter_packages` in `recompress.log`).

The HF revisions above come from the reference check (accessed 2026-11-19), not from the download itself: no file records which revision the local folders were downloaded at. The shard hashes are the binding identifier. `verification/report_G1.md` (item 6) reports that fresh `sha256sum` on the GPU matched all 8 weight files.

### Datasets

| Item | Identifier / revision | Checksum (sha256) | Licence | Source |
|---|---|---|---|---|
| Högskoleprovet, verbal part | Booklets and answer keys from studera.nu (run by UHR), 28 exams 2013-10-26 to 2026-10-18; 93 files (57 booklets, 36 answer keys), fetched 2026-11-19 00:07-00:09 UTC; per-file URL and sha256 in the manifest | `exams/raw/manifest.json` `8214e089633aae4a35c4f52781038f78cb38f748a0be760967ad172c0f268d24` | No reuse licence found; texts include third-party copyrighted material. **Not published** (owner decision 2026-11-19, option (b) in `report/DATA_RIGHTS.md`): the PDFs are git-ignored, only `manifest.json` is in git. Rebuild with `make data` (`bench fetch`) | `exams/raw/manifest.json`; licence: `report/references/article_sources_licences.md` |
| Extracted questions | `exams/questions.jsonl` (1,650 questions). **Not in git** (it holds the question texts). Rebuild with `make data` (`bench fetch` + `bench extract`); `make check-data` checks the rebuilt file against `questions_sha256` in `exams/split.json` | `6d84c5ecdfaec6e585f4743c9e7f06ca06990fd229bc0aeaf452e778a9917955` | as above | computed with `shasum -a 256` on the owner's copy |
| Frozen split | `exams/split.json` (created 2026-11-19T00:55:55Z from `config/split.toml`) | `3c7cc1f0d72928fb3e9a3e5cd739d452a3a8cb98a0eb002d1320a02d0f321eaf` | n/a | computed with `shasum -a 256` |
| Questions hash recorded in split.json | `questions_sha256` field | `6d84c5ec…7955`: **matches** the owner's `questions.jsonl` (and must match a rebuild) | n/a | `exams/split.json`; every `results/runs/*/run.json` also records `questions_sha256`, `questions_sha256_at_freeze` and `split_sha256` with these same values |
| Belebele | HF `facebook/belebele`, revision `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`; configs `swe_Latn`, `eng_Latn`; 500 shared questions | not recorded (ids saved in result files) | CC BY-SA 4.0 | `analysis/other_exams.py` (`BELEBELE_REV`); `revisions` field in `results/experiments/analysis/other_exams/*.json`; licence: `report/references/article_sources_licences.md`, confirmed in `verification/report_G8.md` (4b, 4c) |
| MMLU | HF `cais/mmlu`, revision `c30699e8356da336a370243923dbaf21066bb9fe`; subjects `elementary_mathematics`, `high_school_mathematics`, `college_mathematics` (test split); 400 questions | not recorded (ids saved in result files) | MIT | same files as Belebele (`MMLU_REV`) |
| Riksdagen speeches (damage checks) | Bulk file `https://data.riksdagen.se/dataset/anforande/anforande-202526.json.zip` (session 2025/26), fetched 2026-11-19 | bulk zip `6d3e0bb56427d28b3fe4f3b6c1ebccad9ba4679398219cd3012060630c23f28f` | Riksdagen's terms of use (checked 2026-11-19): free use and redistribution with attribution "Källa: Sveriges riksdag" / "Source: The Swedish Parliament", without implying the Riksdag is behind the work; contains personal data (GDPR). Recorded in `SOURCES.md` and `NOTICE` | `data/damage/SOURCES.md`; `report/references/article_sources_licences.md` ([riksdagen_terms]) |
| Swedish Wikipedia (damage checks) | sv.wikipedia.org plain-text extracts, articles created after the cutoffs | none; the built files contain **0** Wikipedia documents (all 75 text docs and 50 prompts have `source = data.riksdagen.se`) | CC BY-SA 4.0 | `data/damage/SOURCES.md`; counted from `source` field of the two JSONL files |
| Damage text | `data/damage/sv_text.jsonl` (75 docs, 40,154 words) | `6f13e43de548fd58397c600b38b417e92071fcdd01b9e8ae337f11d7c137e912` (recomputed, matches SOURCES.md) | as Riksdagen | `data/damage/SOURCES.md` |
| Damage prompts | `data/damage/sv_prompts.jsonl` (50 prompts) | `81b231ffb94673e9b05422cc111c32f58881d06326f1f3abfbb137ab5695277a` (recomputed, matches) | as Riksdagen | `data/damage/SOURCES.md` |

### Software and hardware

| Item | Version | Source |
|---|---|---|
| llama.cpp | tag `b8680`, commit `15f786e6581598638840276948a7e6183fc96a83`; `llama-server --version` prints `version: 1 (15f786e)`. Licence MIT | `results/logs/recompress.log` (`llama_cpp`); `describe.llama_server_version` in q4 `run.json`; licence and release date (2026-04-06) in `report/references/article_sources_licences.md` |
| llama.cpp CUDA build flags | not recorded | |
| torch (bench runs) | `2.15.0+cu132` | `describe.torch` in all HF `run.json` (all 5 HF standard runs and 138 HF experiment and archive runs agree) |
| transformers | `5.23.0` | `describe.transformers` in HF `run.json`; same in `uv.lock` |
| Other locked packages | torch 2.15.0 (PyPI build; GPU used the `+cu132` build), accelerate 1.16.0, huggingface-hub 2.4.0, safetensors 0.8.0, tokenizers 0.23.3, pymupdf 2.1, numpy 2.4.6 / 2.5.5 (by Python version) | `uv.lock` (sha256 `cbe2197b50657c21a5399200e0f1f1fde2d06cbe7cd64aa29562d2a69f684d30`) |
| `datasets` (Belebele/MMLU loader) | not recorded (run with `uv run --with datasets`, unpinned) | `analysis/other_exams.py` docstring |
| Python | `requires-python = ">=3.11"`; exact version of the GPU bench venv not recorded; converter venv is 3.11 | `pyproject.toml`, `uv.lock`, `scripts/setup_llama_venv.sh` |
| CUDA | only the torch build tag `cu132` (CUDA 13.2); driver version not recorded | `run.json` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | `describe.gpu` in HF `run.json` |
| Code version of each run | not recorded (`git_commit: null`, `git_dirty: null` in every `run.json`) | `results/runs/*/run.json` |

## 2. Random seeds

Scoring uses no randomness. Each answer is one forward pass: the next-token probabilities of the option letters are read, and the highest one is chosen; ties go to the first letter (`src/bench/run.py`, `score_answer`). All free text generation is greedy, so no seed affects it. Seeds matter only for random controls, LoRA-style group training, bootstrap intervals, and a few sampling choices in the editors' own scripts.

| Seed | Value | Where set | Where recorded |
|---|---|---|---|
| Group training (A init, batch order, option shuffles) | `0` | `train_group_delta(seed=0)` default in `src/bench/group_edit.py`; `torch.Generator().manual_seed(seed)` for A, `random.Random(seed)` for batches; `src/bench/group_experiment.py` does not override it | `results/experiments/group/gemma-4b-full/trained/delta.json` (`seed: 0`, `info.training.seed: 0`), `train_result.json`, `results.json` |
| Group random control | `1` | `CONTROL_SEED = 1` in `src/bench/group_experiment.py`; `random_delta_like` in `group_edit.py` | `results/experiments/group/gemma-4b-full/control/control.json` and `delta.json` (`seed: 1`, `control_for: {kind: trained, seed: 0}`) |
| Editor trial random controls | `1` | `CONTROL_SEED = 1` in `src/bench/editor/evaluate.py`; per tensor `manual_seed(stable_seed("control\|1\|<tensor name>"))` | `results/experiments/editor/trial_*/control/control.json` (all four `seed: 1`) |
| Random control in rotation / other-exam analyses | `1` (default of `random_control`) | `analysis/rotation_proof.py`, `analysis/other_exams.py` call `random_control(edit)` | not written to output |
| Single-question edit controls (archived) | per question: `int(sha256("control\|<question id>")[:8], 16)` | `control_seed` / `stable_seed` in `src/bench/experiment.py`; `random_edit_like` in `src/bench/edit.py` | `results/archive/single_question_edits/*/edits/*/control_edit.json` (e.g. `2023-03-25/p5/LAS/13` -> `1426516921`, recomputed and matches); targeted edits have `seed: null` |
| Bootstrap intervals | seed `20260101`, `10000` resamples, percentile interval, paired by question | `DEFAULT_SEED` in `src/bench/stats.py`; samples from `config/decision.toml`. No caller passes another seed (`report.py`, `group_experiment.py`, `editor/evaluate.py`) | not written to output |
| llama.cpp server and perplexity | `--seed 0` (has no effect at temperature 0) | `SEED = 0` in `src/bench/backends/llamacpp.py` | `describe.seed: 0` in q4 `run.json` |
| Dry-run random backend | `0` | `config/models.toml` `[dry-random] seed = 0`; `src/bench/backends/dry.py` | `results/runs/*dry-random*/run.json` |
| Editor exploration scripts (AI editors' own) | `random.seed(0)`, `random.seed(1)`, `random.seed(3)`, `random.Random(seed=0)`; `torch.Generator().manual_seed(0)` / `(1)` for fold splits | `results/editor_runs/gemma_b10/scripts/s03_pass1_read.py` (0), `s10_show.py` (1), `s12_pass2_read.py` (3); `gemma_b25/scripts/lib.py` (0), `s03_read1.py` (0), `s08_cv.py`, `s09_slope.py` (0), `s12_neurons.py`, `s18_posneurons.py` (1); `gemma_b50/scripts/s08_read_prefix.py` (0), `s17_features.py` (1) | scripts only |
| Editor agents (the LLMs that proposed edits) | not recorded | `results/editor_runs/*/run.json` holds only run name, budget, creation time and layout (`dtype: bfloat16`); `parameters.json` is a list of model tensors, not settings | |
| Verification scripts | `R_rescore.py` `SEED = 20261119` (samples of 40, 30, 20 questions); `G4_verify.py` bootstrap seeds 1, 2, 3 and default 12345; `G4_q4_ci_seeds.py` seeds 0-19; `G6_verify.py` 0; `G7_ties_and_averaging.py` 0 | `verification/scripts/` | `verification/scripts/R_results.json` (`seed`) |

`analysis/*.py` set no seeds. Their question sets are fixed lists (whole dev / hidden / test LÄS+MEK sets), and their significance tests are exact binomial / McNemar tests.

Deterministic orders used instead of seeds: the group split sorts dev exam dates by sha256(date) (first round(n/3) = 8 hidden, next 3 validation, the remaining 12 drill; `results/experiments/group/gemma-4b-full/split.json`). The archived dev subset sorts question ids by sha256 (`results/archive/single_question_edits/dev_subset.json`). Belebele (first 500 shared ids) and MMLU (first 400) are also sorted by sha256(id) (`analysis/other_exams.py`). The damage-text documents are sorted by sha256 of their id (`data/damage/SOURCES.md`).

Group training settings (`results/experiments/group/gemma-4b-full/trained/delta.json`): layers 21-24 `mlp.down_proj`, rank 8, lr 3e-5 (AdamW, no weight decay), batch size 8, eval every 25 steps, max steps 2000, patience 10, min right kept 0.95, floor 0.90, shuffle true. It ran 300 steps, stopped on `no_improvement`, and step 50 was chosen. (`train_result.json` records `settings.max_steps: null`, meaning the default 2000 was used.)

## 3. Inference settings

| Setting | Value | Source |
|---|---|---|
| HF dtype / device | bfloat16 on CUDA, in every HF run (the `*-f32` entries exist but no run used them) | `DEFAULT_HF_DTYPE` in `src/bench/backends/__init__.py`; `describe.dtype` in all HF `run.json` |
| Prompt | Chat template from each model's HF tokenizer folder: `apply_chat_template(add_generation_prompt=True, enable_thinking=False)`. The q4 backends use the same HF tokenizer and send token ids, so both precisions read identical ids | `src/bench/prompts.py`; `config/models.toml` (`tokenizer`); `src/bench/backends/llamacpp.py` |
| Letter scoring (HF) | softmax (float32) of the last-position logits; a letter's probability is the sum over all vocabulary tokens whose stripped text is that letter; `prob_correct` is normalised over the option letters | `src/bench/backends/hf.py`; `src/bench/run.py` |
| Letter scoring (llama.cpp) | `/completion`, `n_predict 1`, `temperature 0`, `n_probs 20`, `cache_prompt false`; letters outside the top 20 count as 0 | `src/bench/backends/llamacpp.py`; `top_probs: 20` in q4 `run.json` |
| llama-server flags | `-c 4096 -ngl all --parallel 1 --seed 0 --host 127.0.0.1` | `src/bench/backends/llamacpp.py`; `ctx_size: 4096`, `n_gpu_layers: "all"` in q4 `run.json` |
| Perplexity | HF: non-overlapping windows of 1024 tokens per text. llama.cpp: `llama-perplexity -c 1024 -ngl all --seed 0` on the joined texts. The two are not comparable (`verification/report_G1.md`) | `WINDOW` in `hf.py`; `PERPLEXITY_CTX` in `llamacpp.py` |
| Damage generation | greedy (`do_sample=False` / `temperature 0`), max 200 new tokens, raw prompt text with no chat template; HF generates in left-padded batches of 64. Loop = some 4-word sequence appears 3 or more times | `src/bench/damage.py`; `src/bench/backends/hf.py` |
| Edit layers | Qwen layer 21, Gemma layer 23 (`mlp.down_proj`, 0-based) | `config/models.toml` |
| Splits | test: 2025-10-19, 2026-04-18, 2026-10-18 (180 questions); gemma_extra: 2024-10-20, 2025-04-05 (120); dev: all other exams (1,350) | `config/split.toml`; `n_questions` in `run.json` |

## 4. Decision rule (`config/decision.toml`)

| Parameter | Value | Meaning |
|---|---|---|
| `metric` | `prob_correct` | mean probability on the correct option, test split |
| `confidence` | 0.95 | CI level for rules 1 and 2 |
| `bootstrap_samples` | 10000 | paired bootstrap resamples (seed 20260101, `src/bench/stats.py`) |
| Rule 1 | CI of (edited - baseline) entirely above 0 | score goes up |
| Rule 2 | CI of (edit gain - random-control gain) entirely above 0 | beats a random edit of the same size |
| `max_perplexity_increase` | 0.05 | perplexity may rise at most 5% (relative) |
| `max_loop_rate_increase` | 2.0 | loop rate may rise at most 2 percentage points |
| `headline_sections` | `["LAS", "MEK"]` | only these decide rules 1 and 2; ORD is reported separately |

Config file checksums: `config/decision.toml` `ec3d0d35…f17`, `config/models.toml` `f9d88228…a8d`, `config/split.toml` `8bbeebd4…d21`, `pyproject.toml` `86d7426f…497`.
