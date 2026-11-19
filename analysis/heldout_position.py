"""Held-out position test and inference-time alternatives (hidden papers + test exams, LÄS+MEK only).

Conditions: unedited; editor 3's weight change (trial_b50_k); unedited + prompt note; unedited +
option averaging (each option's probability averaged over the 4 rotations). Writes per-question
results to experiments/analysis/heldout_position.jsonl and a summary to heldout_position.json."""
import json, collections, dataclasses
from pathlib import Path
from bench.backends import make_backend
from bench.records import load_questions
from bench.prompts import build_prompt, render_prompt
from bench.editor.evaluate import load_edit, applied, edit_source
from bench import paths
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py

NOTE = ("Obs: svarsalternativen står i slumpmässig ordning. Läs alla alternativ noga "
        "innan du väljer, även det första.")
out_dir = RESULTS / "experiments/analysis"; out_dir.mkdir(parents=True, exist_ok=True)
group_split = json.load(open(RESULTS / "experiments/group/gemma-4b-full/split.json"))["question_ids"]
hidden_ids = set(group_split["hidden_wrong"]) | set(group_split["hidden_right"])
test_ids = set(json.load(open(paths.SPLIT))["splits"]["test"]["question_ids"])
qs = {q.id: q for q in load_questions(paths.QUESTIONS)}
sets = {"hidden": [qs[i] for i in sorted(hidden_ids) if qs[i].section in ("LAS", "MEK")],
        "test": [qs[i] for i in sorted(test_ids) if qs[i].section in ("LAS", "MEK")]}
backend = make_backend("gemma-4b-full"); tok = backend.tokenizer_path

def probs(q, note=False):
    msgs = build_prompt(q)
    if note:
        msgs = [dict(m) for m in msgs]; msgs[-1]["content"] = NOTE + "\n\n" + msgs[-1]["content"]
    p = backend.letter_probabilities(render_prompt(msgs, tok), list("ABCD"), q)
    s = sum(p.get(L, 0.0) for L in "ABCD") or 1.0
    return {L: p.get(L, 0.0) / s for L in "ABCD"}

def rotations(q):
    """Probabilities per option TEXT for each of the 4 rotations, and correctness per answer position."""
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

records, summary = [], {}
def run(condition, edit=None):
    for set_name, questions in sets.items():
        res = collections.Counter(); pos = collections.defaultdict(list)
        for q in questions:
            orig = probs(q); orig_ok = max("ABCD", key=orig.get) == q.answer
            right_at, avg_ok = rotations(q)
            row = {"condition": condition, "set": set_name, "id": q.id, "section": q.section, "answer": q.answer,
                   "original_order_correct": orig_ok, "option_average_correct": avg_ok, "correct_by_position": right_at}
            if condition == "unedited":
                note = probs(q, note=True); row["prompt_note_correct"] = max("ABCD", key=note.get) == q.answer
                res["prompt_note"] += row["prompt_note_correct"]
            records.append(row)
            res["original"] += orig_ok; res["option_average"] += avg_ok
            res["all4"] += all(right_at.values()); res["none4"] += not any(right_at.values())
            res["only_A_wrong"] += (not right_at["A"]) and all(right_at[L] for L in "BCD")
            res["only_A_right"] += right_at["A"] and not any(right_at[L] for L in "BCD")
            for L, ok in right_at.items(): pos[L].append(ok)
        n = len(questions)
        summary[f"{condition}/{set_name}"] = {"n": n, **{k: v for k, v in res.items()},
            "accuracy_by_answer_position": {L: round(100 * sum(v) / n, 1) for L, v in sorted(pos.items())}}
        print(condition, set_name, json.dumps(summary[f"{condition}/{set_name}"]), flush=True)

run("unedited")
with applied(backend, load_edit(edit_source("trial_b50_k"))):
    run("editor3_edit")
backend.close()
with open(out_dir / "heldout_position.jsonl", "w") as f:
    for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump({"note_prompt": NOTE, "summary": summary}, open(out_dir / "heldout_position.json", "w"), indent=2, ensure_ascii=False)
