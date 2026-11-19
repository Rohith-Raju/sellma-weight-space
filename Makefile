# Reproduce the main experiment and re-check the saved results.  Full guide: REPRODUCE.md
#
#   make help            list the targets
#
# Needs: uv; for anything that runs a model, an NVIDIA GPU (models run in bfloat16) and the model
# weights in models/; for 4-bit runs, llama.cpp b8680 built at /root/llama.cpp (or set
# `llama_cpp_bin` for the model in config/models.toml).

SHELL    := /bin/bash
.SHELLFLAGS := -o pipefail -c
UV       ?= uv
PY       := $(UV) run python
# Revisions of the model weights (see CONFIG.md). The shard checksums in results/runs/*/run.json are
# the binding check: `make check-models`.
GEMMA_REPO := google/gemma-3-4b-it
GEMMA_REV  := 093f9f388b31de276ce2de164bdc2081324b9767
QWEN_REPO  := Qwen/Qwen3.5-9B
QWEN_REV   := c202236235762e1c871ad0ccb60c8ee5ba337b9a

# Verification scripts that need only saved files (they were written for the owner's Mac copy and
# read <project>/gpu_results/...; `make verify` points them at this checkout and results/ instead).
# Most also read exams/questions.jsonl, which is not in git (run `make data` first). G3_claims also
# reads failure_cases.jsonl, which holds question text and exists only in the full GPU copy. A script
# whose ignored input is missing is skipped with a message (see VERIFY_CHECK below).
VERIFY_OFFLINE := G1_data_split G1_runs G2_rotation_counts G2_verify G3_claims G3_inventory \
                  G4_q4_ci_seeds G4_verify G6_q4_diff G6_verify G7_ties_and_averaging G7_verify G8_recompute
# These also read the exam PDFs in exams/raw/ (make fetch).
VERIFY_PDF     := G1_answer_keys G1_booklet_text G1_visual_keys
# Read-only analysis scripts whose printed output is saved in results/experiments/analysis/reports/.
REPORTS        := letters habit_q4 edited_habit group_cmp proof_check oe_stats a_picks a_habit_all_tests

.PHONY: help env test fetch extract data check-data models check-models compress standard-runs \
        prepare-main main rotation-proof heldout other-exams check-main reports verify verify-pdf ensure-data \
        figures report

help:
	@echo "env            install the environment (uv sync --extra gpu)"
	@echo "test           run the unit tests (no GPU needed)"
	@echo "data           fetch + extract the exams, then check-data (the split itself is frozen in exams/split.json)"
	@echo "check-data     questions.jsonl / split.json checksums vs the saved runs"
	@echo "models         download Gemma 3 4B it and Qwen3.5-9B at the recorded revisions into models/"
	@echo "check-models   weight files and our 4-bit GGUFs vs the checksums in results/runs/*/run.json"
	@echo "compress       make our own Q4_K_M GGUFs with llama.cpp b8680"
	@echo "standard-runs  the 10 standard runs of the unedited models (GPU)"
	@echo "main           the main experiment: rotation test, held-out alternatives, other exams (GPU)"
	@echo "check-main     compare a fresh 'make main' with the saved results"
	@echo "reports        re-run the read-only analyses on results/ and diff against the saved reports"
	@echo "verify         re-run the independent verification scripts on results/ (no GPU; needs make data)"
	@echo "verify-pdf     3 more verification scripts that read the exam PDFs (needs make fetch)"
	@echo "figures        redraw the report figures from the saved results (report/figures/)"
	@echo "report         build report/report.pdf from report/report_full.md (pandoc + xelatex)"

# --- environment and data ------------------------------------------------------------------------

env:
	$(UV) sync --extra gpu

test:
	$(UV) run pytest

fetch:
	$(UV) run bench fetch

extract:
	$(UV) run bench extract

# `bench split` is NOT run: exams/split.json is frozen (it stores the checksum of questions.jsonl and
# `bench split` refuses to overwrite it). check-data confirms a rebuilt questions.jsonl matches it.
data: fetch extract check-data

check-data:
	$(PY) scripts/check_inputs.py data

# --- models --------------------------------------------------------------------------------------

# Gemma is gated on Hugging Face: accept its terms and run `uv run hf auth login` first.
models:
	$(UV) run hf download $(GEMMA_REPO) --revision $(GEMMA_REV) --local-dir models/gemma-3-4b-it-official
	$(UV) run hf download $(QWEN_REPO) --revision $(QWEN_REV) --local-dir models/Qwen3.5-9B-HF

check-models:
	$(PY) scripts/check_inputs.py models

# Needs llama.cpp b8680 at /root/llama.cpp and its converter venv at /root/llama-venv
# (scripts/setup_llama_venv.sh); see REPRODUCE.md.
compress:
	$(PY) -m bench.compress models/gemma-3-4b-it-official models/compressed/gemma-4b-full-Q4_K_M.gguf --quant Q4_K_M
	$(PY) -m bench.compress models/Qwen3.5-9B-HF models/compressed/qwen-9b-full-Q4_K_M.gguf --quant Q4_K_M

# --- the 10 standard runs (each writes runs/<time>_<model>_<split>/) --------------------------------

standard-runs:
	$(UV) run bench run qwen-9b-full  --split dev
	$(UV) run bench run qwen-9b-full  --split test
	$(UV) run bench run qwen-9b-q4    --split dev
	$(UV) run bench run qwen-9b-q4    --split test
	$(UV) run bench run gemma-4b-full --split dev
	$(UV) run bench run gemma-4b-full --split test
	$(UV) run bench run gemma-4b-full --split gemma_extra
	$(UV) run bench run gemma-4b-q4   --split dev
	$(UV) run bench run gemma-4b-q4   --split test
	$(UV) run bench run gemma-4b-q4   --split gemma_extra

# --- the main experiment -----------------------------------------------------------------------------
# Outputs go under SELMA_RESULTS (default: this folder): logs/rotation_proof.json and
# experiments/analysis/{heldout_position.*,other_exams/}. They never overwrite results/.

# Put the saved inputs where the scripts and the bench look for them (only if not already there):
#   editor 3's change      results/editor_runs/trial_b50_k/ -> editor_runs/trial_b50_k/
#   the group split        results/experiments/group/gemma-4b-full/split.json -> experiments/group/...
#                          (defines the 8 hidden papers; rebuilt by `uv run bench group split --model gemma-4b-full`)
#   the first rotation run results/logs/rotation_test.jsonl -> logs/ (rotation_proof.py compares to it)
prepare-main:
	mkdir -p editor_runs experiments/group/gemma-4b-full logs
	[ -e editor_runs/trial_b50_k ] || cp -R results/editor_runs/trial_b50_k editor_runs/trial_b50_k
	[ -e experiments/group/gemma-4b-full/split.json ] || cp results/experiments/group/gemma-4b-full/split.json experiments/group/gemma-4b-full/split.json
	[ -e logs/rotation_test.jsonl ] || cp results/logs/rotation_test.jsonl logs/rotation_test.jsonl

main: prepare-main rotation-proof heldout other-exams check-main

rotation-proof: prepare-main
	$(PY) analysis/rotation_proof.py | tee logs/rotation_proof.out

heldout: prepare-main
	$(PY) analysis/heldout_position.py

# Belebele and MMLU come from Hugging Face at pinned revisions (inside the script); `datasets` is
# not a bench dependency, so it is added for this command only.
other-exams: prepare-main
	$(UV) run --with datasets python analysis/other_exams.py unedited
	$(UV) run --with datasets python analysis/other_exams.py editor3
	$(UV) run --with datasets python analysis/other_exams.py random

check-main:
	$(PY) scripts/check_main.py

# --- re-checking the saved results (no GPU) -------------------------------------------------------------

reports:
	@mkdir -p work/reports; for s in $(REPORTS); do \
	  SELMA_RESULTS=results $(PY) analysis/$$s.py > work/reports/$$s.txt 2>&1 || exit 1; \
	  if cmp -s work/reports/$$s.txt results/experiments/analysis/reports/$$s.txt; \
	  then echo "same       $$s"; else echo "DIFFERENT  $$s (see work/reports/$$s.txt)"; fi; \
	done

# The verification scripts find the repo from their own location and read gpu_results/ when present,
# otherwise results/ (see the header of each script), so they run unchanged on any machine.

# Inputs that are git-ignored because they hold exam text. Before each script, VERIFY_CHECK looks for
# the input names in the script; if the script reads one that is missing here, the script is skipped
# with a message (and listed at the end) instead of stopping the whole loop.
#   exams/questions.jsonl  rebuilt by `make data`
#   exams/raw/             rebuilt by `make fetch` (part of `make data`)
#   failure_cases.jsonl    only in the full GPU copy (gpu_results/ on the owner's Mac); never published
FC_FILE := results/archive/single_question_edits/gemma-4b-full/failure_cases.jsonl
VERIFY_CHECK = f=verification/scripts/$$s.py; miss=""; \
	  if grep -q 'questions.jsonl' $$f && [ ! -e exams/questions.jsonl ]; then miss="exams/questions.jsonl (run make data)"; fi; \
	  if grep -qE '/raw|pymupdf' $$f && [ ! -d exams/raw/2013-10-26 ]; then miss="$$miss exams/raw/ PDFs (run make fetch)"; fi; \
	  if grep -q 'failure_cases.jsonl' $$f && [ ! -e $(FC_FILE) ]; then miss="$$miss $(FC_FILE) (full GPU copy only; holds question text, not published)"; fi

define VERIFY_LOOP
	@skipped=""; for s in $(1); do echo "== $$s"; $(VERIFY_CHECK); \
	  if [ -n "$$miss" ]; then echo "SKIPPED $$s: needs $$miss"; skipped="$$skipped $$s"; continue; fi; \
	  $(PY) verification/scripts/$$s.py || exit 1; done; \
	if [ -n "$$skipped" ]; then echo "== skipped, missing git-ignored input:$$skipped (see REPRODUCE.md section 8)"; fi
endef

# The exam questions are not in the repo (data rights). If they are missing, rebuild them first from the
# public PDFs on studera.nu (make data: fetch, extract, and check against the checksum in exams/split.json).
ensure-data:
	@if [ ! -f exams/questions.jsonl ]; then echo "exams/questions.jsonl is missing: rebuilding it with make data"; $(MAKE) data; fi

verify: ensure-data
	$(call VERIFY_LOOP,$(VERIFY_OFFLINE))

verify-pdf: ensure-data
	$(call VERIFY_LOOP,$(VERIFY_PDF))

# --- figures and the report PDF ---------------------------------------------------------------------

# Reads only saved files (gpu_results/experiments/analysis/ if that folder exists, else
# results/experiments/analysis/) and rewrites report/figures/fig*.png and fig*.svg.
figures:
	$(UV) run --with matplotlib python -I report/figures/make_figures.py

# Needs pandoc, xelatex and the fonts Charter, Helvetica Neue and Menlo (macOS). Writes report/report.pdf.
report:
	bash report/latex/build.sh
