"""Where the analysis scripts read and write.

The scripts were first run on the GPU machine from /root/sellma, with the bench's outputs (runs/,
experiments/, logs/, editor_runs/) next to src/. That path used to be written into each script; it is
now configurable, and nothing else about what the scripts compute has changed.

  SELMA_ROOT     the project folder (holds src/, config/, exams/, models/). Default: this repository.
  SELMA_RESULTS  the folder that holds runs/, experiments/, logs/ and editor_runs/. Default: SELMA_ROOT,
                 which is right on a machine that ran the experiments. To re-read the saved results that
                 ship with the repository, set SELMA_RESULTS=results (relative paths are taken from
                 SELMA_ROOT).

Scripts that load the model also use the bench package, which finds questions, the split and editor
change files (editor_runs/<run>/final_edit.*) relative to the repository itself (src/bench/paths.py).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SELMA_ROOT") or Path(__file__).resolve().parents[1])
RESULTS = Path(os.environ.get("SELMA_RESULTS") or ROOT)
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS
