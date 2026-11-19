"""Is editor 3's position effect real? Rotation test (dev LÄS+MEK, each question x4 positions) under:
unedited x2, editor change, random control (same tensors/pattern/norm), unedited again after removal.
Saves per-question correctness and margins; prints noise, paired exact McNemar tests, comparison to first run."""
import json, collections, dataclasses, math
from bench.backends import make_backend
from bench.records import load_questions
from bench.prompts import build_prompt, render_prompt
from bench.editor.evaluate import load_edit, applied, edit_source, random_control
from bench import paths
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py

split = json.load(open(paths.SPLIT))["splits"]["dev"]["question_ids"]
qs = {q.id: q for q in load_questions(paths.QUESTIONS)}
questions = [qs[i] for i in split if qs[i].section in ("LAS", "MEK")]
backend = make_backend("gemma-4b-full"); tok = backend.tokenizer_path
dup = sum(1 for q in questions if len(set(q.options[L] for L in "ABCD")) < 4)

def run(label):
    res = {}
    for q in questions:
        texts = [q.options[L] for L in "ABCD"]; correct = q.options[q.answer]; row = {}; marg = {}
        for shift in range(4):
            rotated = texts[shift:] + texts[:shift]; ans = "ABCD"[rotated.index(correct)]
            rq = dataclasses.replace(q, options=dict(zip("ABCD", rotated)), answer=ans)
            p = backend.letter_probabilities(render_prompt(build_prompt(rq), tok), list("ABCD"), rq)
            row[ans] = max("ABCD", key=lambda L: p.get(L, 0.0)) == ans
            marg[ans] = p.get(ans, 0.0) - max(p.get(L, 0.0) for L in "ABCD" if L != ans)
        res[q.id] = {"ok": row, "margin": marg}
    print("done", label, flush=True)
    return res

R = {}
R["unedited_1"] = run("unedited_1")
R["unedited_2"] = run("unedited_2")
edit = load_edit(edit_source("trial_b50_k"))
with applied(backend, edit): R["editor3"] = run("editor3")
with applied(backend, random_control(edit)): R["random"] = run("random")
R["unedited_after"] = run("unedited_after")
backend.close()
(RESULTS / "logs").mkdir(parents=True, exist_ok=True)
json.dump(R, open(RESULTS / "logs/rotation_proof.json", "w"))

def mcnemar(b, c):  # exact two-sided binomial on discordant pairs
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
ids = [q.id for q in questions]
def summary(name):
    r = R[name]; pos = {L: 100 * sum(r[i]["ok"][L] for i in ids) / len(ids) for L in "ABCD"}
    allr = sum(all(r[i]["ok"].values()) for i in ids); none = sum(not any(r[i]["ok"].values()) for i in ids)
    return pos, allr, none
print(f"\nquestions {len(ids)}; questions with duplicate option texts {dup}")
for name in R:
    pos, allr, none = summary(name)
    print(f"{name:15} A {pos['A']:.1f} B {pos['B']:.1f} C {pos['C']:.1f} D {pos['D']:.1f} | all4 {allr} none4 {none} depends {len(ids)-allr-none}")

def flips(a, b):
    return sum(R[a][i]["ok"][L] != R[b][i]["ok"][L] for i in ids for L in "ABCD")
print("\nanswer flips (of 3600 question-positions): unedited_1 vs unedited_2", flips("unedited_1", "unedited_2"),
      "| unedited_1 vs unedited_after", flips("unedited_1", "unedited_after"),
      "| unedited vs editor3", flips("unedited_1", "editor3"), "| unedited vs random", flips("unedited_1", "random"))
try:
    first = {json.loads(l)["id"]: json.loads(l) for l in open(RESULTS / "logs/rotation_test.jsonl")}
    f = sum(first[i][L] != R["unedited_1"][i]["ok"][L] for i in ids for L in "ABCD")
    fa = sum(all(first[i][L] for L in "ABCD") for i in ids)
    print(f"first saved run vs unedited_1: {f} flips; first run all4 = {fa}")
except Exception as e: print("first run compare failed", e)

for name in ("editor3", "random", "unedited_2"):
    for metric, fn in (("right at A", lambda r, i: r[i]["ok"]["A"]),
                       ("consistent (right in all 4)", lambda r, i: all(r[i]["ok"].values()))):
        b = sum(fn(R[name], i) and not fn(R["unedited_1"], i) for i in ids)
        c = sum(not fn(R[name], i) and fn(R["unedited_1"], i) for i in ids)
        print(f"{name:11} {metric:28} gained {b:3} lost {c:3} net {b-c:+4}  exact McNemar p = {mcnemar(b, c):.2g}")
near = sum(abs(R["unedited_1"][i]["margin"][L]) < 0.01 for i in ids for L in "ABCD")
print(f"\nquestion-positions within 1 point of a tie (unedited): {near}")
