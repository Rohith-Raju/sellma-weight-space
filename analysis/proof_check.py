import json
from math import comb
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
d = json.load(open(RESULTS / "logs/rotation_proof.json"))
def p2(a, b):
    n = a + b; k = min(a, b); return min(1, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
ids = list(d["unedited_1"])
diff = sum(d["unedited_1"][q]["ok"][L] != d[k][q]["ok"][L] for k in ("unedited_2", "unedited_after") for q in ids for L in "ABCD")
print("questions", len(ids), "| answers differing between the three unedited repeats:", diff)
u = d["unedited_1"]
for name in ("editor3", "random"):
    e = d[name]
    gained = sum((not u[q]["ok"]["A"]) and e[q]["ok"]["A"] for q in ids); lost = sum(u[q]["ok"]["A"] and not e[q]["ok"]["A"] for q in ids)
    all4g = sum((not all(u[q]["ok"].values())) and all(e[q]["ok"].values()) for q in ids)
    all4l = sum(all(u[q]["ok"].values()) and not all(e[q]["ok"].values()) for q in ids)
    print(f"{name}: right at A gained {gained} lost {lost} (p={p2(gained, lost):.2g}) | right in all 4 gained {all4g} lost {all4l} (p={p2(all4g, all4l):.2g})")
print("unedited always right:", sum(all(u[q]['ok'].values()) for q in ids), "never:", sum(not any(u[q]['ok'].values()) for q in ids))
