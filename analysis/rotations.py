"""Show every 4-option dev question to unedited Gemma 4 times, rotating the options so the
correct answer sits once in each position. Same questions in every position, so any accuracy
difference between positions can only come from position, not from question difficulty."""
import json, collections, dataclasses
from bench.backends import make_backend
from bench.records import load_questions
from bench.prompts import build_prompt, render_prompt
from bench import paths
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py

split = json.load(open(paths.SPLIT))["splits"]["dev"]["question_ids"]
qs = {q.id: q for q in load_questions(paths.QUESTIONS)}
questions = [qs[i] for i in split if qs[i].section in ("LAS", "MEK")]
backend = make_backend("gemma-4b-full")
tok = backend.tokenizer_path
(RESULTS / "logs").mkdir(parents=True, exist_ok=True)
out = open(RESULTS / "logs/rotation_test.jsonl", "w")
by_pos = collections.defaultdict(list)
for q in questions:
    texts = [q.options[L] for L in "ABCD"]
    correct_text = q.options[q.answer]
    row = {"id": q.id, "section": q.section}
    for shift in range(4):
        rotated = texts[shift:] + texts[:shift]
        options = dict(zip("ABCD", rotated))
        answer = "ABCD"[rotated.index(correct_text)]
        rq = dataclasses.replace(q, options=options, answer=answer)
        probs = backend.letter_probabilities(render_prompt(build_prompt(rq), tok), list("ABCD"), rq)
        chosen = max("ABCD", key=lambda L: probs.get(L, 0.0))
        row[answer] = chosen == answer
        by_pos[answer].append(chosen == answer)
    out.write(json.dumps(row) + "\n")
backend.close()
print("questions:", len(questions))
for L in "ABCD":
    v = by_pos[L]; print(f"correct answer shown at {L}: right {sum(v)}/{len(v)} = {100*sum(v)/len(v):.1f}%")
rows = [json.loads(l) for l in open(RESULTS / "logs/rotation_test.jsonl")]
a_only_wrong = sum(1 for r in rows if not r["A"] and r["B"] and r["C"] and r["D"])
a_only_right = sum(1 for r in rows if r["A"] and not r["B"] and not r["C"] and not r["D"])
always = sum(1 for r in rows if all(r[L] for L in "ABCD"))
never = sum(1 for r in rows if not any(r[L] for L in "ABCD"))
print(f"right in all 4 positions: {always}; wrong in all 4: {never}")
print(f"right everywhere EXCEPT when the answer is at A: {a_only_wrong}; right ONLY at A: {a_only_right}")
