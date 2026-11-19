import json
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
base = f"{RESULTS}/experiments/group/gemma-4b-full"
split = json.load(open(f"{base}/split.json"))["question_ids"]
def ans(c, s):
    return {r["question_id"]: r for r in map(json.loads, open(f"{base}/{c}/{s}/answers.jsonl"))}
hw, hr = set(split["hidden_wrong"]), set(split["hidden_right"])
for c in ["baseline", "trained", "control"]:
    h = ans(c, "hidden"); t = ans(c, "test")
    fixed = sum(h[q]["is_correct"] for q in hw); kept = sum(h[q]["is_correct"] for q in hr)
    hl = [a for a in t.values() if a["section"] in ("LAS", "MEK")]
    acc = 100 * sum(a["is_correct"] for a in hl) / len(hl); p = sum(a["prob_correct"] for a in hl) / len(hl)
    print(f"{c:9} hidden: wrong fixed {fixed}/117, right kept {kept}/363 | test headline {acc:.1f}% p={p:.3f}")
