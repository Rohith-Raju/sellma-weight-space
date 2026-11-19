"""Independent verification of claim group G2. Reads raw answers.jsonl only."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, math
from collections import Counter
from fractions import Fraction
from pathlib import Path

ROOT = Path(_REPO + "")
RUNS = Path(_RES) / "runs"
FULL = RUNS / "20261119T060704Z_gemma-4b-full_dev" / "answers.jsonl"
Q4 = RUNS / "20261119T061105Z_gemma-4b-q4_dev" / "answers.jsonl"

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

full, q4 = load(FULL), load(Q4)
questions = {q["id"]: q for q in load(ROOT / "exams" / "questions.jsonl")}
split = json.load(open(ROOT / "exams" / "split.json"))
dev_ids = split["splits"]["dev"]["question_ids"]

print("== integrity ==")
for name, rows in (("full", full), ("q4", q4)):
    ids = [r["question_id"] for r in rows]
    print(name, "rows", len(rows), "unique", len(set(ids)), "ids == dev split:", set(ids) == set(dev_ids))
    print("  key matches questions.jsonl:", sum(r["correct_letter"] == questions[r["question_id"]]["answer"] for r in rows))
    print("  section matches:", sum(r["section"] == questions[r["question_id"]]["section"] for r in rows))
    print("  chosen == argmax(letter_probs):", sum(r["chosen_letter"] == max(r["letter_probs"], key=r["letter_probs"].get) for r in rows))
    print("  is_correct consistent:", sum(r["is_correct"] == (r["chosen_letter"] == r["correct_letter"]) for r in rows))
    print("  chosen letters:", sorted(Counter(r["chosen_letter"] for r in rows).items(), key=lambda kv: str(kv[0])))
    print("  sections:", Counter(r["section"] for r in rows))

def pct(a, b):
    return f"{a}/{b} = {100*a/b:.2f}%"

def table(rows, label):
    print(f"\n== {label} (n={len(rows)}) ==")
    key = Counter(r["correct_letter"] for r in rows)
    pick = Counter(r["chosen_letter"] for r in rows)
    print(" key counts :", {k: key[k] for k in "ABCD"}, "other:", {k: v for k, v in key.items() if k not in "ABCD"})
    print(" pick counts:", {k: pick[k] for k in "ABCD"}, "other:", {k: v for k, v in pick.items() if k not in "ABCD"})
    for k in "ABCD":
        sub = [r for r in rows if r["correct_letter"] == k]
        print(f" acc key={k}:", pct(sum(r['chosen_letter'] == k for r in sub), len(sub)))
    a = [r for r in rows if r["correct_letter"] == "A"]
    o = [r for r in rows if r["correct_letter"] != "A"]
    print(" acc key=A  :", pct(sum(r['is_correct'] for r in a), len(a)))
    print(" acc key=BCD:", pct(sum(r['is_correct'] for r in o), len(o)))
    print(" overall    :", pct(sum(r['is_correct'] for r in rows), len(rows)))
    print(" key=A wrong picks:", dict(Counter(r["chosen_letter"] for r in a if not r["is_correct"])))
    return a, o

hl = lambda rows: [r for r in rows if r["section"] in ("LAS", "MEK")]
a, o = table(hl(full), "CLAIM 1: full, dev LAS+MEK")

def ztest(x1, n1, x2, n2):
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    z_pooled = (p1 - p2) / math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z_unpooled = (p1 - p2) / math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return z_pooled, z_unpooled

xa, na = sum(r["is_correct"] for r in a), len(a)
xo, no = sum(r["is_correct"] for r in o), len(o)
zp, zu = ztest(xa, na, xo, no)
print(f"\n== CLAIM 4 z: key=A {xa}/{na} vs others {xo}/{no}: pooled z={zp:.4f}, unpooled z={zu:.4f}")
a4, o4 = table(hl(q4), "q4, dev LAS+MEK (for wording check)")
zp4, zu4 = ztest(sum(r["is_correct"] for r in a4), len(a4), sum(r["is_correct"] for r in o4), len(o4))
print(f" q4 z: pooled {zp4:.4f}, unpooled {zu4:.4f}")

print("\n== CLAIM 2: per section, full vs q4 ==")
for sec in ("MEK", "LAS", "ORD"):
    for name, rows in (("full", full), ("q4", q4)):
        table([r for r in rows if r["section"] == sec], f"{name} {sec}")

print("\n== CLAIM 3: agreement full vs q4 ==")
f = {r["question_id"]: r for r in full}
g = {r["question_id"]: r for r in q4}
assert set(f) == set(g)
same = sum(f[i]["chosen_letter"] == g[i]["chosen_letter"] for i in f)
print(" same letter:", same, "of", len(f))
for sec in ("ORD", "LAS", "MEK"):
    ids = [i for i in f if f[i]["section"] == sec]
    print(f"  {sec}:", sum(f[i]['chosen_letter'] == g[i]['chosen_letter'] for i in ids), "of", len(ids))
ids = [i for i in f if f[i]["section"] in ("LAS", "MEK")]
print("  LAS+MEK:", sum(f[i]['chosen_letter'] == g[i]['chosen_letter'] for i in ids), "of", len(ids))
# paired A-habit comparison full vs q4 on key=A headline questions
ka = [i for i in ids if f[i]["correct_letter"] == "A"]
b = sum(f[i]["is_correct"] and not g[i]["is_correct"] for i in ka)
c = sum(g[i]["is_correct"] and not f[i]["is_correct"] for i in ka)

def sign_p(x, y):
    """Exact two-sided binomial sign test, p=1/2 (symmetric => 2 * smaller tail, capped at 1)."""
    n, k = x + y, min(x, y)
    tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2 ** n)
    return float(min(Fraction(1), 2 * tail))

print(f" key=A LAS+MEK paired: full-right/q4-wrong {b}, q4-right/full-wrong {c}, sign p={sign_p(b, c):.4g}")
kb = [i for i in ids if f[i]["correct_letter"] != "A"]
b2 = sum(f[i]["is_correct"] and not g[i]["is_correct"] for i in kb)
c2 = sum(g[i]["is_correct"] and not f[i]["is_correct"] for i in kb)
print(f" key=BCD LAS+MEK paired: full-right/q4-wrong {b2}, q4-right/full-wrong {c2}, sign p={sign_p(b2, c2):.4g}")

print("\n== CLAIM 4: exact two-sided sign tests ==")
for x, y in ((22, 4), (22, 2), (123, 9), (124, 9)):
    print(f" {x} vs {y}: p = {sign_p(x, y):.6g}")
