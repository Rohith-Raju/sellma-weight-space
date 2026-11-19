"""Which option does Gemma pick in each of the four orders? Rotation test that keeps the choice.

Same rotation as rotation_proof.py (each LÄS/MEK question asked four times, options shifted one place each
time, so every option is at A once), but it saves the probability Gemma gives each option *text* in every
order, so a wrong answer can be told apart from a position guess. Unedited Gemma and Gemma with editor 3's
weight edit (trial_b50_k), on dev (900, includes the 8 hidden papers) and test (120).

Output: experiments/analysis/rotation_choices.json
  {condition: {question_id: {"split", "section", "answer": original letter of the right option,
                             "orders": {position of the right answer ("A".."D"):
                                        {"probs": {original option letter: probability},
                                         "chosen": original option letter,
                                         "chosen_position": letter it was shown at}}}}}
"""
import json, dataclasses, time
from bench.backends import make_backend
from bench.records import load_questions
from bench.prompts import build_prompt, render_prompt
from bench.editor.evaluate import load_edit, applied, edit_source
from bench import paths
from _paths import RESULTS

splits = json.load(open(paths.SPLIT))["splits"]
qs = {q.id: q for q in load_questions(paths.QUESTIONS)}
questions = [(s, qs[i]) for s in ("dev", "test") for i in splits[s]["question_ids"] if qs[i].section in ("LAS", "MEK")]
backend = make_backend("gemma-4b-full"); tok = backend.tokenizer_path


def run(label):
    res, t0 = {}, time.time()
    for split, q in questions:
        orig = list("ABCD")  # original letters, in original order
        orders = {}
        for shift in range(4):
            shown = orig[shift:] + orig[:shift]  # shown[k] = original letter displayed at position "ABCD"[k]
            rq = dataclasses.replace(q, options={"ABCD"[k]: q.options[shown[k]] for k in range(4)},
                                     answer="ABCD"[shown.index(q.answer)])
            p = backend.letter_probabilities(render_prompt(build_prompt(rq), tok), list("ABCD"), rq)
            probs = {shown[k]: float(p.get("ABCD"[k], 0.0)) for k in range(4)}
            pos_chosen = max("ABCD", key=lambda L: p.get(L, 0.0))
            orders[rq.answer] = {"probs": probs, "chosen": shown["ABCD".index(pos_chosen)],
                                 "chosen_position": pos_chosen}
        res[q.id] = {"split": split, "section": q.section, "answer": q.answer, "orders": orders}
    print(f"done {label}: {len(res)} questions in {time.time() - t0:.0f}s", flush=True)
    return res


R = {"unedited": run("unedited")}
with applied(backend, load_edit(edit_source("trial_b50_k"))):
    R["editor3"] = run("editor3")
backend.close()
out = RESULTS / "experiments/analysis/rotation_choices.json"
json.dump(R, open(out, "w"))
print("wrote", out)

# Sanity check against the saved rotation test (dev): right/wrong per position must be identical.
old = json.load(open(RESULTS / "experiments/analysis/rotation_proof.json"))
for cond, oldkey in (("unedited", "unedited_1"), ("editor3", "editor3")):
    diff = sum((R[cond][i]["orders"][L]["chosen"] == R[cond][i]["answer"]) != old[oldkey][i]["ok"][L]
               for i in old[oldkey] for L in "ABCD")
    print(f"{cond}: differences from rotation_proof.json on dev: {diff} of {4 * len(old[oldkey])}")
