import json
from math import comb
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
D = f"{RESULTS}/experiments/analysis/other_exams"
r = {c: json.load(open(f"{D}/{c}.json"))["results"] for c in ("unedited", "editor3", "random")}
def p2(a, b):
    n = a + b; k = min(a, b); return 1.0 if n == 0 else min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
for s in ("belebele_sv", "belebele_en", "mmlu_math"):
    u = r["unedited"][s]; ids = list(u); n = len(ids)
    avg = lambda d: sum(sum(d[q].values()) for q in ids) / (4 * n) * 100
    print(f"== {s} (n={n}) mean accuracy over 4 positions: unedited {avg(u):.1f}%, editor3 {avg(r['editor3'][s]):.1f}%, random {avg(r['random'][s]):.1f}%")
    for c in ("editor3", "random"):
        e = r[c][s]
        for L in "ABCD":
            g = sum((not u[q][L]) and e[q][L] for q in ids); l = sum(u[q][L] and not e[q][L] for q in ids)
            if g or l: print(f"   {c:8} answer at {L}: gained {g:3d} lost {l:3d} (p={p2(g, l):.2g})")
