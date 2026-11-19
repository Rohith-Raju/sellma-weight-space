"""(a) exact ties in rotation_proof.json (max() breaks ties toward the earlier letter);
(b) test of heldout_position.py's option-averaging logic (function body copied verbatim, fake model)."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, collections, dataclasses, random
R = json.load(open(_RES + "/experiments/analysis/rotation_proof.json"))
for n, r in R.items():
    t = [(i, L, v["ok"][L]) for i, v in r.items() for L in "ABCD" if v["margin"][L] == 0]
    odd = [(i, L) for i, v in r.items() for L in "ABCD" if (v["margin"][L] > 0) != v["ok"][L] and v["margin"][L] != 0]
    print(n, "exact ties:", len(t), collections.Counter((L, ok) for _, L, ok in t), "| non-tie ok/margin disagreements:", len(odd))

@dataclasses.dataclass
class Q: options: dict; answer: str
def make_rotations(probs):
    def rotations(q):  # verbatim from gpu_results/analysis/heldout_position.py
        texts = [q.options[L] for L in "ABCD"]; correct = q.options[q.answer]
        per_text = collections.defaultdict(list); right_at = {}
        for shift in range(4):
            rotated = texts[shift:] + texts[:shift]
            rq = dataclasses.replace(q, options=dict(zip("ABCD", rotated)), answer="ABCD"[rotated.index(correct)])
            p = probs(rq)
            for L, t in zip("ABCD", rotated): per_text[t].append(p[L])
            right_at[rq.answer] = max("ABCD", key=p.get) == rq.answer
        averaged_choice = max(texts, key=lambda t: sum(per_text[t]))
        return right_at, averaged_choice == correct
    return rotations
rng = random.Random(0); bad = 0; cover = collections.Counter()
for trial in range(20000):
    texts = [f"t{k}" for k in range(4)]; q = Q(dict(zip("ABCD", texts)), rng.choice("ABCD"))
    table = {}
    def probs(rq):
        key = tuple(rq.options[L] for L in "ABCD")
        if key not in table:
            w = [rng.random() for _ in range(4)]; s = sum(w); table[key] = {L: x / s for L, x in zip("ABCD", w)}
        return table[key]
    right_at, avg_ok = make_rotations(probs)(q)
    # independent reference: enumerate the 4 cyclic orders directly
    tot = collections.Counter(); ref_right = {}
    for s in range(4):
        order = texts[s:] + texts[:s]; p = table[tuple(order)]
        for pos, t in enumerate(order): tot[t] += p["ABCD"[pos]]
        cpos = order.index(q.options[q.answer]); ref_right["ABCD"[cpos]] = max(range(4), key=lambda k: p["ABCD"[k]]) == cpos
    ref_avg = max(texts, key=lambda t: tot[t]) == q.options[q.answer]
    cover[len(right_at)] += 1
    bad += (ref_avg != avg_ok) or (ref_right != right_at) or len(table) != 4 or any(abs(sum(1 for _ in v) - 0) < 0 for v in [])
print("simulated questions 20000; disagreements with independent reference:", bad, "| positions covered per question:", dict(cover))
