"""G9 independent check: single-question rank-1 edits (archive/single_question_edits).

  - "9 of 12 made the test score clearly worse": paired bootstrap 95% CI of mean prob_correct
    (edited - baseline) over the 120 test LAS+MEK questions; CI entirely below zero.
    Own pure-Python bootstrap, 10,000 resamples, run with 5 different seeds, plus a normal-theory interval.
  - test headline accuracy per edit / control, letter shares.
  - weights changed per edit = entries_changed (recorded fraction of the matrix whose bf16 value changed)
    x matrix size (matrix size read from the u, v shapes in targeted_edit.safetensors headers).
Run: cd <repo> && python3 verification/scripts/G9_single.py
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import glob
import json
import math
import random
import struct
from collections import Counter

B = "results/archive/single_question_edits/"
TEST = set(json.load(open("exams/split.json"))["splits"]["test"]["question_ids"])


def load(path):
    rows = {}
    for l in open(path + "/answers.jsonl"):
        r = json.loads(l)
        rows[r["question_id"]] = r
    return rows


def headline(rows):
    return {q: r for q, r in rows.items() if r["section"] in ("LAS", "MEK")}


def boot(diffs, seed, n=10000):
    rng = random.Random(seed)
    k = len(diffs)
    means = []
    for _ in range(n):
        s = 0.0
        for _ in range(k):
            s += diffs[rng.randrange(k)]
        means.append(s / k)
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def st_header(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    return json.loads(f.read(n))


TOTAL = {"gemma-4b-full": 4_300_079_472, "qwen-9b-full": 9_653_104_368}
n_worse = 0
n_edits = 0
sizes = {}
for m in ("gemma-4b-full", "qwen-9b-full"):
    base = headline(load(B + m + "/baseline/full_test"))
    assert set(base) <= TEST and len(base) == 120, (m, len(base))
    bacc = 100 * sum(r["is_correct"] for r in base.values()) / 120
    bA = Counter(r["chosen_letter"] for r in base.values())
    print(f"\n== {m}: baseline test headline {bacc:.1f}%  letter shares " +
          " ".join(f"{L} {100*bA[L]/120:.0f}%" for L in "ABCD"))
    sizes[m] = []
    for e in sorted(glob.glob(B + m + "/edits/*")):
        s = json.load(open(e + "/search.json"))
        if not s.get("fixed"):
            print(f"  {e.split('/')[-1]:24s} {s['status']}")
            continue
        n_edits += 1
        ed = headline(load(e + "/targeted_test"))
        ct = headline(load(e + "/control_test"))
        assert set(ed) == set(base) == set(ct)
        qs = sorted(base)
        d = [ed[q]["prob_correct"] - base[q]["prob_correct"] for q in qs]
        dc = [ed[q]["prob_correct"] - ct[q]["prob_correct"] for q in qs]
        mean = sum(d) / len(d)
        sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (len(d) - 1))
        cis = [boot(d, seed) for seed in (1, 2, 3, 4, 5)]
        hi_max = max(c[1] for c in cis)
        lo_min = min(c[0] for c in cis)
        worse = all(c[1] < 0 for c in cis)
        n_worse += worse
        acc = 100 * sum(r["is_correct"] for r in ed.values()) / 120
        cacc = 100 * sum(r["is_correct"] for r in ct.values()) / 120
        let = Counter(r["chosen_letter"] for r in ed.values())
        # size
        h = st_header(e + "/targeted_edit.safetensors")
        mat = h["u"]["shape"][0] * h["v"]["shape"][0]
        t = [t for t in s["tried"] if t["alpha_fraction"] == s["edit"]["alpha_fraction"]][0]
        ac = json.load(open(e + "/targeted_applied_change.json"))[0]["entries_changed"]
        ch = t["entries_changed"] * mat
        sizes[m].append(ch)
        print(f"  {e.split('/')[-1]:24s} alpha {s['edit']['alpha_fraction']:.2f} test {acc:.1f}% (control {cacc:.1f}%) "
              f"dprob {mean:+.4f} CI[{lo_min:+.4f},{hi_max:+.4f}] (normal {mean-1.96*sd/math.sqrt(120):+.4f},{mean+1.96*sd/math.sqrt(120):+.4f}) "
              f"worse={worse} | vs control {sum(dc)/120:+.4f} | letters " + " ".join(f"{L}{100*let[L]/120:.0f}" for L in "ABCD")
              + f" | changed {ch:,.0f} ({100*ch/TOTAL[m]:.3f}%) matrix {h['u']['shape']}x{h['v']['shape']} applied==search {abs(ac - t['entries_changed'])<1e-12}")
    print(f"  size range {min(sizes[m])/1e6:.2f}M - {max(sizes[m])/1e6:.2f}M = "
          f"{100*min(sizes[m])/TOTAL[m]:.3f}% - {100*max(sizes[m])/TOTAL[m]:.3f}%")
print(f"\nedits: {n_edits}; CI entirely below zero (all 5 seeds): {n_worse}")

# group edit
d = json.load(open("results/experiments/group/gemma-4b-full/trained/delta.json"))
mat = 2560 * 10240
hdr = st_header("results/experiments/group/gemma-4b-full/trained/delta.safetensors")
print("group delta tensors:", {k: v["shape"] for k, v in hdr.items() if k != "__metadata__"})
ch = sum(L["entries_changed"] * mat for L in d["info"]["layers"])
print(f"group edit: {ch:,.0f} changed = {100*ch/TOTAL['gemma-4b-full']:.3f}%  (rank {d['rank']}, layers {d['layers']})")

# trial_b50_k
j = json.load(open("results/editor_runs/trial_b50_k/final_edit.json"))
sz = j["size"]
print(f"trial_b50_k: {sz['weights_changed']:,} of {sz['model_weights']:,} = {100*sz['weights_changed']/sz['model_weights']:.4f}%; "
      f"touched {sz['weights_touched']:,} = 256 x 2560 = {256*2560:,}")
h = st_header("results/editor_runs/trial_b50_k/final_edit.safetensors")
print("trial_b50_k tensors:", {k: v.get("shape") for k, v in h.items() if k != "__metadata__"})
# count non-zero values in the stored delta (pure python, f32)
k = [k for k in h if k.startswith("values:")][0]
off = h[k]["data_offsets"]
f = open("results/editor_runs/trial_b50_k/final_edit.safetensors", "rb")
n = struct.unpack("<Q", f.read(8))[0]
f.seek(8 + n + off[0])
vals = struct.unpack(f"<{(off[1]-off[0])//4}f", f.read(off[1] - off[0]))
print("trial_b50_k stored delta: values", len(vals), "non-zero", sum(v != 0 for v in vals))
rk = [k for k in h if k.startswith("rows:")][0]
off = h[rk]["data_offsets"]
f.seek(8 + n + off[0])
rows = struct.unpack(f"<{(off[1]-off[0])//8}q", f.read(off[1] - off[0]))
print("rows:", min(rows), "-", max(rows), "count", len(rows))
print(f"q4 weight check internal: 314,961/366,231 = {100*314961/366231:.2f}%")
print(f"Astra 131,072 / 0.016% -> implied model size {131072/0.00016:,.0f}")
