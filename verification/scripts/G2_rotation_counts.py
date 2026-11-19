"""Side check for G2.4: where do '123 vs 9' / '124 vs 9' come from? (rotation data, belongs to G7)"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json
d = json.load(open(_RES + "/experiments/analysis/rotation_proof.json"))
print("top-level keys:", list(d))
for run, qs in d.items():
    if not isinstance(qs, dict):
        continue
    rows = [v["ok"] for v in qs.values() if isinstance(v, dict) and "ok" in v]
    if not rows:
        continue
    wrong_only_a = sum((not o["A"]) and o["B"] and o["C"] and o["D"] for o in rows)
    right_only_a = sum(o["A"] and not (o["B"] or o["C"] or o["D"]) for o in rows)
    print(run, "n", len(rows), "wrong-only-at-A", wrong_only_a, "right-only-at-A", right_only_a,
          "A-right", sum(o["A"] for o in rows))
