import json, os
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
E = f"{RESULTS}/experiments"
def stats(path):
    if not os.path.exists(path): return None
    rows = [r for r in map(json.loads, open(path)) if r["section"] in ("LAS", "MEK")]
    a = [r for r in rows if r["correct_letter"] == "A"]; o = [r for r in rows if r["correct_letter"] != "A"]
    return (100*sum(r["is_correct"] for r in rows)/len(rows), 100*sum(r["is_correct"] for r in a)/len(a),
            100*sum(r["is_correct"] for r in o)/len(o), sum(r["chosen_letter"] == "A" for r in rows), len(a), len(rows))
def show(label, s):
    print(f"{label:34} " + ("missing" if not s else f"all {s[0]:5.1f}% | answer=A right {s[1]:5.1f}% | answer=B/C/D right {s[2]:5.1f}% | picked A {s[3]:3d} times (A correct {s[4]}) n={s[5]}"))
for split in ("hidden", "test"):
    print(f"===== {split} papers (LAS+MEK)")
    show("unedited, full", stats(f"{E}/group/gemma-4b-full/baseline/{split}/answers.jsonl"))
    q4b = f"{E}/group/gemma-4b-full/baseline/q4_hidden/answers.jsonl" if split == "hidden" else None
    if split == "test":
        import glob; q4b = sorted(glob.glob(f"{RESULTS}/runs/*_gemma-4b-q4_test"))[-1] + "/answers.jsonl"
    show("unedited, 4-bit", stats(q4b))
    for run in ("trial_b10_q_full", "trial_b10_q_half", "trial_b25_mlp", "trial_b50_k"):
        show(f"{run}, full", stats(f"{E}/editor/{run}/edited/{split}/answers.jsonl"))
        show(f"{run}, 4-bit", stats(f"{E}/editor/{run}/edited/q4_{split}/answers.jsonl"))
        show(f"{run}, random ctrl full", stats(f"{E}/editor/{run}/control/{split}/answers.jsonl"))
