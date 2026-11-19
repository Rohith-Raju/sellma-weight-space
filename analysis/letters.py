import json, glob, collections
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
run = sorted(glob.glob(f"{RESULTS}/runs/*_gemma-4b-full_dev"))[-1]
rows = [r for r in map(json.loads, open(run + "/answers.jsonl")) if r["section"] in ("LAS", "MEK")]
key = collections.Counter(r["correct_letter"] for r in rows)
chosen = collections.Counter(r["chosen_letter"] for r in rows)
print("n =", len(rows))
for L in "ABCD":
    sub = [r for r in rows if r["correct_letter"] == L]
    ok = sum(r["is_correct"] for r in sub)
    print(f"{L}: right answer {key[L]:3d} times | Gemma picked it {chosen[L]:3d} times | right when the answer is {L}: {ok}/{len(sub)} = {100*ok/len(sub):.0f}%")
wrongA = [r for r in rows if r["correct_letter"] == "A" and not r["is_correct"]]
print("when the answer was A and Gemma got it wrong, it picked:", dict(collections.Counter(r["chosen_letter"] for r in wrongA)))
