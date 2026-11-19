import json, collections
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
D = f"{RESULTS}/experiments/analysis/other_exams"
r = {c: json.load(open(f"{D}/{c}_with_choices.json"))["results"] for c in ("unedited", "editor3")}
for s in ("belebele_sv", "belebele_en", "mmlu_math"):
    print(f"== {s}")
    for c in ("unedited", "editor3"):
        picks = r[c][s + "/chosen"]; n = 4 * len(picks)
        cnt = collections.Counter(L for q in picks.values() for L in q.values())
        print(f"   {c:8} picks (of {n}, each letter correct {n//4} times): " + "  ".join(f"{L} {cnt[L]:4d} ({100*cnt[L]/n:4.1f}%)" for L in "ABCD"))
