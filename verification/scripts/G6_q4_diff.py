"""G6: do the 4-bit edited runs actually answer differently from the unedited 4-bit run?"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json
from pathlib import Path
G = Path(_RES + "")
def load(d): return {json.loads(l)["question_id"]: json.loads(l) for l in open(d / "answers.jsonl")}
base = {"q4_hidden": load(G / "experiments/group/gemma-4b-full/baseline/q4_hidden"),
        "q4_test": load(G / "runs/20261119T061642Z_gemma-4b-q4_test")}
for t in ["trial_b50_k", "trial_b10_q_full", "trial_b25_mlp", "trial_b10_q_half"]:
    for s, b in base.items():
        e = load(G / "experiments/editor" / t / "edited" / s)
        nd = sum(e[q]["chosen_letter"] != b[q]["chosen_letter"] for q in b)
        dp = sum(abs(e[q]["prob_correct"] - b[q]["prob_correct"]) > 1e-6 for q in b)
        print(f"{t:18s} {s:10s} chosen differs {nd:3d}/{len(b)}  prob_correct differs {dp}/{len(b)}")
