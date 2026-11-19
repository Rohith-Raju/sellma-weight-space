"""G8 verifier: recompute every other-exams number from the raw per-question JSONs.

Independent code (does not import oe_stats.py / a_picks.py / a_habit_all_tests.py).
Run: uv run python -I verification/scripts/G8_recompute.py
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json
from collections import Counter
from math import comb
from pathlib import Path

ROOT = Path(_RES + "/experiments/analysis")
OE = ROOT / "other_exams"
SETS = ("belebele_sv", "belebele_en", "mmlu_math")
L4 = "ABCD"


def load(name):
    return json.loads((OE / f"{name}.json").read_text())


def binom_two_sided(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


runs = {c: load(c) for c in ("unedited", "editor3", "random")}
wc = {c: load(f"{c}_with_choices") for c in ("unedited", "editor3")}

print("== metadata")
for name, d in list(runs.items()) + [(k + "_with_choices", v) for k, v in wc.items()]:
    print(f"  {name:26} condition={d['condition']:9} revisions={d['revisions']}")

print("\n== integrity: each question has exactly A,B,C,D booleans")
for name, d in list(runs.items()) + [(k + "_wc", v) for k, v in wc.items()]:
    for s in SETS:
        bad = [q for q, v in d["results"][s].items() if sorted(v) != list(L4) or not all(isinstance(x, bool) for x in v.values())]
        assert not bad, (name, s, bad[:3])
print("  ok")

print("\n== question ids identical across conditions and runs")
for s in SETS:
    ids = [list(d["results"][s]) for d in list(runs.values()) + list(wc.values())]
    print(f"  {s}: n={len(ids[0])}, unique={len(set(ids[0]))}, same id list (order too) in all 5 files: {all(i == ids[0] for i in ids)}")
sv_ids, en_ids = set(runs["unedited"]["results"]["belebele_sv"]), set(runs["unedited"]["results"]["belebele_en"])
print(f"  belebele sv ids == en ids: {sv_ids == en_ids} (|sv|={len(sv_ids)}, |en|={len(en_ids)})")

print("\n== *_with_choices vs first runs (correctness per question per position)")
for c in ("unedited", "editor3"):
    for s in SETS:
        a, b = runs[c]["results"][s], wc[c]["results"][s]
        diff = [(q, L) for q in a for L in L4 if a[q][L] != b[q][L]]
        print(f"  {c:8} {s:12}: cells differing {len(diff)} of {4 * len(a)}" + (f"  e.g. {diff[:5]}" if diff else ""))

print("\n== *_with_choices internal: ok[L] == (chosen[L] == L)")
for c in ("unedited", "editor3"):
    for s in SETS:
        ok, ch = wc[c]["results"][s], wc[c]["results"][s + "/chosen"]
        assert set(ok) == set(ch)
        bad = [(q, L) for q in ok for L in L4 if ok[q][L] != (ch[q][L] == L)]
        badletters = [(q, L) for q in ch for L in L4 if ch[q][L] not in L4]
        print(f"  {c:8} {s:12}: inconsistent {len(bad)}, invalid letters {len(badletters)}")

print("\n== claim 1: accuracy by answer position, all-4 count")
for c in ("unedited", "editor3", "random"):
    for s in SETS:
        r = runs[c]["results"][s]; n = len(r)
        acc = {L: 100 * sum(v[L] for v in r.values()) / n for L in L4}
        all4 = sum(all(v.values()) for v in r.values())
        mean = sum(acc.values()) / 4
        print(f"  {c:8} {s:12} n={n}: " + " ".join(f"{L} {acc[L]:.2f}" for L in L4) + f" | all4 {all4} | mean {mean:.2f}")

print("\n== claim 1 (random vs unedited): max |diff| per position, cells differing")
for s in SETS:
    u, r = runs["unedited"]["results"][s], runs["random"]["results"][s]; n = len(u)
    d = {L: 100 * (sum(r[q][L] for q in u) - sum(u[q][L] for q in u)) / n for L in L4}
    cells = sum(u[q][L] != r[q][L] for q in u for L in L4)
    print(f"  {s:12}: diffs " + " ".join(f"{L} {d[L]:+.2f}" for L in L4) + f" | cells differing {cells}")

print("\n== claim 2: gains/losses (editor3 vs unedited), sign test, means")
for c in ("editor3", "random"):
    for s in SETS:
        u, e = runs["unedited"]["results"][s], runs[c]["results"][s]
        parts = []
        for L in L4:
            g = sum(e[q][L] and not u[q][L] for q in u)
            l = sum(u[q][L] and not e[q][L] for q in u)
            parts.append(f"{L} +{g}/-{l} (p={binom_two_sided(g, l):.2g})")
        mu = 100 * sum(sum(v.values()) for v in u.values()) / (4 * len(u))
        me = 100 * sum(sum(v.values()) for v in e.values()) / (4 * len(e))
        print(f"  {c:8} {s:12}: " + "  ".join(parts) + f" | mean {mu:.2f} -> {me:.2f}")

print("\n== claim 3: letter picks (% of all 4n answers), from *_with_choices")
for c in ("unedited", "editor3"):
    for s in SETS:
        ch = wc[c]["results"][s + "/chosen"]; N = 4 * len(ch)
        cnt = Counter(x for v in ch.values() for x in v.values())
        print(f"  {c:8} {s:12} (N={N}): " + " ".join(f"{L} {cnt[L]} ({100 * cnt[L] / N:.2f})" for L in L4))

print("\n== claim 5: wrong-only-at-L vs right-only-at-L on every test (unedited)")
tests = {}
proof = json.loads((ROOT / "rotation_proof.json").read_text())
print("  rotation_proof.json keys:", list(proof)[:10])
tests["HP dev"] = {q: v["ok"] for q, v in proof["unedited_1"].items()}
held = [json.loads(l) for l in (ROOT / "heldout_position.jsonl").read_text().splitlines() if l.strip()]
print("  heldout conditions/sets:", Counter((r["condition"], r["set"]) for r in held))
for s in ("hidden", "test"):
    tests[f"HP {s}"] = {r["id"]: r["correct_by_position"] for r in held if r["condition"] == "unedited" and r["set"] == s}
for s in SETS:
    tests[s] = runs["unedited"]["results"][s]
for name, data in tests.items():
    n = len(data)
    out = []
    for L in L4:
        others = [M for M in L4 if M != L]
        wo = sum((not v[L]) and all(v[M] for M in others) for v in data.values())
        ro = sum(v[L] and not any(v[M] for M in others) for v in data.values())
        out.append(f"{L} {wo}/{ro} (p={binom_two_sided(wo, ro):.2g})")
    acc = " ".join(f"{L} {100 * sum(v[L] for v in data.values()) / n:.1f}" for L in L4)
    print(f"  {name:12} n={n}: acc {acc} | wrong-only/right-only: " + "  ".join(out))
