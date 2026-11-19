import json, glob, collections
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
runs = {"full": sorted(glob.glob(f"{RESULTS}/runs/*_gemma-4b-full_dev"))[-1],
        "4-bit": sorted(glob.glob(f"{RESULTS}/runs/*_gemma-4b-q4_dev"))[-1]}
answers = {k: {r["question_id"]: r for r in map(json.loads, open(v + "/answers.jsonl"))} for k, v in runs.items()}
for name, ans in answers.items():
    for sec in ("MEK", "LAS"):
        rows = [a for a in ans.values() if a["section"] == sec]
        key_a = [a for a in rows if a["correct_letter"] == "A"]
        other = [a for a in rows if a["correct_letter"] != "A"]
        chosen_a = sum(a["chosen_letter"] == "A" for a in rows)
        print(f"{name:5} {sec}: right when key=A {sum(a['is_correct'] for a in key_a)}/{len(key_a)} "
              f"({100*sum(a['is_correct'] for a in key_a)/len(key_a):.0f}%), key=B/C/D "
              f"{100*sum(a['is_correct'] for a in other)/len(other):.0f}% | A chosen {chosen_a} times, A is key {len(key_a)} times")
same = sum(answers["full"][q]["chosen_letter"] == answers["4-bit"][q]["chosen_letter"] for q in answers["full"])
print(f"same letter chosen by full and 4-bit: {same}/{len(answers['full'])}")
