"""Summarise the 4-bit rotation test (q4_rotation.py shards) next to the same test at full precision
(heldout_position.jsonl): accuracy by answer position, position-dependence, paired gains/losses at A."""
import glob, json
from math import comb
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "gpu_results" if (ROOT / "gpu_results").exists() else ROOT
A = DATA / "experiments/analysis"
OUT_DIR = A / "reports"   # under DATA (the repository root on a clone), so results/ is never overwritten
if not (A / "q4_rotation").exists():   # fresh clone: read the saved copy in results/
    A = ROOT / "results/experiments/analysis"

def p2(a, b):
    n = a + b; k = min(a, b); return 1.0 if n == 0 else min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def load_q4(which):
    rows = {}
    for f in glob.glob(str(A / f"q4_rotation/{which}_shard*of4.jsonl")):
        for r in map(json.loads, open(f)): rows[(r["set"], r["id"])] = r["correct_by_position"]
    return rows

full = {}
for r in map(json.loads, open(A / "heldout_position.jsonl")):
    full.setdefault(r["condition"], {})[(r["set"], r["id"])] = r["correct_by_position"]
conds = {"full unedited": full["unedited"], "full editor 3": full["editor3_edit"],
         "4-bit unedited": load_q4("unedited"), "4-bit editor 3": load_q4("edited"), "4-bit random control": load_q4("control")}
lines = []
for s in ("hidden", "test"):
    lines.append(f"== {s}")
    for name, d in conds.items():
        keys = [k for k in d if k[0] == s]; n = len(keys)
        if not n:
            lines.append(f"   {name:22} missing"); continue
        pos = "  ".join(f"{L} {100*sum(d[k][L] for k in keys)/n:5.1f}" for L in "ABCD")
        always = sum(all(d[k].values()) for k in keys); never = sum(not any(d[k].values()) for k in keys)
        only_a = sum((not d[k]["A"]) and all(d[k][L] for L in "BCD") for k in keys)
        lines.append(f"   {name:22} n={n}  by position: {pos} | all 4: {always} | none: {never} | wrong only at A: {only_a}")
    for base, new in (("full unedited", "full editor 3"), ("4-bit unedited", "4-bit editor 3"), ("4-bit unedited", "4-bit random control")):
        b, e = conds[base], conds[new]
        keys = [k for k in b if k[0] == s and k in e]
        if not keys: continue
        g = sum((not b[k]["A"]) and e[k]["A"] for k in keys); l = sum(b[k]["A"] and not e[k]["A"] for k in keys)
        tot_b = sum(sum(b[k].values()) for k in keys); tot_e = sum(sum(e[k].values()) for k in keys)
        lines.append(f"   {new} vs {base}: right at A gained {g} lost {l} (p={p2(g, l):.2g}); "
                     f"all positions {tot_b} -> {tot_e} of {4*len(keys)}")
out = OUT_DIR / "q4_rotation_report.txt"; out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n"); print("\n".join(lines))
