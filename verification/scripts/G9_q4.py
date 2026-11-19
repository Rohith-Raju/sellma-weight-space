"""G9 independent check: 440-question compression counts (full precision vs 4-bit, rotation test).

Reads raw per-question files only:
  results/experiments/analysis/heldout_position.jsonl     full precision, hidden 320 + test 120
  results/experiments/analysis/q4_rotation/*_shard*.jsonl 4-bit (unedited / edited / control)
  results/experiments/analysis/rotation_choices.json      cross-check of full precision dev(hidden)/test
Run: cd <repo> && python3 verification/scripts/G9_q4.py
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import glob
import json
from collections import Counter

A = "results/experiments/analysis/"
P = "ABCD"

full = {}
for l in open(A + "heldout_position.jsonl"):
    r = json.loads(l)
    full.setdefault(r["condition"], {})[(r["set"], r["id"])] = r
q4 = {}
for w in ("unedited", "edited", "control"):
    d = {}
    files = sorted(glob.glob(A + f"q4_rotation/{w}_shard*of4.jsonl"))
    for f in files:
        for l in open(f):
            r = json.loads(l)
            assert (r["set"], r["id"]) not in d, "duplicate"
            d[(r["set"], r["id"])] = r
    q4[w] = d
    print(w, "files", len(files), "rows", len(d), Counter(k[0] for k in d), Counter(r["section"] for r in d.values()))

C = {"full_unedited": full["unedited"], "full_edited": full["editor3_edit"],
     "q4_unedited": q4["unedited"], "q4_edited": q4["edited"], "q4_control": q4["control"]}
keys = sorted(full["unedited"])
for n, d in C.items():
    assert set(d) == set(keys), n
print("all conditions cover the same", len(keys), "questions")

# cross-check full precision with rotation_choices
RC = json.load(open(A + "rotation_choices.json"))
mism = 0
for (s, q) in keys:
    for cond, rc in (("full_unedited", RC["unedited"]), ("full_edited", RC["editor3"])):
        for p in P:
            ok = rc[q]["orders"][p]["chosen"] == rc[q]["answer"]
            if ok != C[cond][(s, q)]["correct_by_position"][p]:
                mism += 1
print("full-precision heldout_position vs rotation_choices mismatches:", mism)

# 4-bit: is chosen_by_position consistent with correct_by_position (chosen position == position of answer)?
inc = sum((r["chosen_by_position"][p] == p) != r["correct_by_position"][p] for d in q4.values() for r in d.values() for p in P)
print("4-bit chosen/correct inconsistencies:", inc)


def cnt(d, ks):
    return {p: sum(d[k]["correct_by_position"][p] for k in ks) for p in P}


for label, ks in (("all 440", keys), ("hidden", [k for k in keys if k[0] == "hidden"]), ("test", [k for k in keys if k[0] == "test"])):
    print(f"\n== {label} ({len(ks)})")
    for n, d in C.items():
        c = cnt(d, ks)
        only_a = sum((not d[k]["correct_by_position"]["A"]) and all(d[k]["correct_by_position"][p] for p in "BCD") for k in ks)
        print(f"  {n:14s} " + " ".join(f"{p} {c[p]}" for p in P) + f" total {sum(c.values())}/{4*len(ks)} wrong-only-at-A {only_a}")
    cu, cq = cnt(C["full_unedited"], ks), cnt(C["q4_unedited"], ks)
    print("  full unedited -> 4-bit unedited per position:", {p: cq[p] - cu[p] for p in P})
    for base, new in (("full_unedited", "full_edited"), ("q4_unedited", "q4_edited"), ("q4_unedited", "q4_control")):
        b, e = C[base], C[new]
        g = sum((not b[k]["correct_by_position"]["A"]) and e[k]["correct_by_position"]["A"] for k in ks)
        l = sum(b[k]["correct_by_position"]["A"] and not e[k]["correct_by_position"]["A"] for k in ks)
        print(f"  at A {new} vs {base}: fixed {g} broken {l}")

# option-A pick rates at 4-bit (chosen_by_position = displayed letter chosen)
print("\n== 4-bit option-A pick rates")
for label, ks in (("all 440", keys), ("hidden", [k for k in keys if k[0] == "hidden"]), ("test", [k for k in keys if k[0] == "test"])):
    for n in ("q4_unedited", "q4_edited", "q4_control"):
        d = C[n]
        pr = sum(d[k]["chosen_by_position"]["A"] == "A" for k in ks)
        pw = sum(d[k]["chosen_by_position"][p] == "A" for k in ks for p in "BCD")
        print(f"  {label:8s} {n:12s} A right->A {100*pr/len(ks):.1f}%  A wrong->A {100*pw/(3*len(ks)):.1f}%  "
              f"picks of A right {100*pr/(pr+pw):.1f}% ({pr}/{pr+pw})")
# same at full precision on the 440 (from rotation_choices positions)
for nm, rc in (("full_unedited", RC["unedited"]), ("full_edited", RC["editor3"])):
    pr = sum(rc[q]["orders"]["A"]["chosen_position"] == "A" for _, q in keys)
    pw = sum(rc[q]["orders"][p]["chosen_position"] == "A" for _, q in keys for p in "BCD")
    print(f"  all 440  {nm:12s} picks of A right {100*pr/(pr+pw):.1f}% ({pr}/{pr+pw})")
