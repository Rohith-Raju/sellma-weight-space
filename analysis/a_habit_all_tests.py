"""Does 'Gemma overlooks option A' hold on every test? For each test (unedited Gemma, rotation data):
accuracy by answer position, and a sign test on questions right everywhere except A vs right only at A,
plus the same comparison for every other letter (is A special, or does each test have its own weak letter?)."""
import json
from math import comb
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py

R = f"{RESULTS}/experiments/analysis"
def p2(a, b):
    n = a + b; k = min(a, b); return 1.0 if n == 0 else min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

tests = {}
proof = json.load(open(f"{R}/rotation_proof.json"))["unedited_1"]
tests["Högskoleprovet dev (900)"] = {q: v["ok"] for q, v in proof.items()}
held = [json.loads(l) for l in open(f"{R}/heldout_position.jsonl")]
for s, label in (("hidden", "Högskoleprovet hidden (320)"), ("test", "Högskoleprovet test 2025-26 (120)")):
    tests[label] = {r["id"]: r["correct_by_position"] for r in held if r["condition"] == "unedited" and r["set"] == s}
oe = json.load(open(f"{R}/other_exams/unedited.json"))["results"]
for s, label in (("belebele_sv", "Belebele Swedish (500)"), ("belebele_en", "Belebele English, same questions (500)"), ("mmlu_math", "MMLU maths (400)")):
    tests[label] = oe[s]

for label, data in tests.items():
    n = len(data)
    acc = {L: 100 * sum(v[L] for v in data.values()) / n for L in "ABCD"}
    only_wrong = {L: sum((not v[L]) and all(v[M] for M in "ABCD" if M != L) for v in data.values()) for L in "ABCD"}
    only_right = {L: sum(v[L] and not any(v[M] for M in "ABCD" if M != L) for v in data.values()) for L in "ABCD"}
    print(f"== {label}")
    print("   right by answer position: " + "  ".join(f"{L} {acc[L]:.1f}%" for L in "ABCD"))
    for L in "ABCD":
        print(f"   {L}: wrong only at {L} {only_wrong[L]:3d} | right only at {L} {only_right[L]:3d} | p = {p2(only_wrong[L], only_right[L]):.2g}")
