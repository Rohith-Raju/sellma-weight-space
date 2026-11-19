"""G9 spot-checks of older numbers in RESEARCH_NOTE.md and report/report_full.md, from raw files.
Run: cd <repo> && python3 verification/scripts/G9_older.py
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import glob
import json
from collections import Counter
from math import comb, prod

R = "results/"
SPLIT = json.load(open("exams/split.json"))


def answers(path):
    return {json.loads(l)["question_id"]: json.loads(l) for l in open(path)}


def hl(rows):
    return {q: r for q, r in rows.items() if r["section"] in ("LAS", "MEK")}


def acc(rows):
    return 100 * sum(r["is_correct"] for r in rows.values()) / len(rows)


def p_sign(a, b):  # exact two-sided binomial (McNemar) p
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


print("== split")
for s, v in SPLIT["splits"].items():
    ids = v["question_ids"]
    dates = sorted({q.split("/")[0] for q in ids})
    print(f"  {s}: {len(ids)} questions, {len(dates)} papers, {dates[0]} .. {dates[-1]}; LAS+MEK {sum(q.split('/')[2] in ('LAS','MEK') for q in ids)}")

print("\n== standard runs, headline (LAS+MEK)")
for f in sorted(glob.glob(R + "runs/2026*/answers.jsonl")):
    if "dry" in f:
        continue
    rows = hl(answers(f))
    print(f"  {f.split('/')[2]:40s} n={len(rows)} {acc(rows):.1f}%")

print("\n== group edit (hidden 8 papers, full precision)")
G = R + "experiments/group/gemma-4b-full/"
gs = json.load(open(G + "split.json"))
for split in ("hidden", "test"):
    b = answers(G + f"baseline/{split}/answers.jsonl")
    t = answers(G + f"trained/{split}/answers.jsonl")
    c = answers(G + f"control/{split}/answers.jsonl") if glob.glob(G + f"control/{split}/answers.jsonl") else None
    bh, th = hl(b), hl(t)
    fixed = sum((not bh[q]["is_correct"]) and th[q]["is_correct"] for q in bh)
    broken = sum(bh[q]["is_correct"] and not th[q]["is_correct"] for q in bh)
    nw = sum(not r["is_correct"] for r in bh.values())
    print(f"  {split}: n={len(bh)} baseline {acc(bh):.1f}% trained {acc(th):.1f}% fixed {fixed}/{nw} broken {broken}/{len(bh)-nw} net {fixed-broken} p={p_sign(fixed, broken):.2f}"
          + (f" control {acc(hl(c)):.1f}%" if c else ""))
    # all sections too (split counts 117/363 may include ORD)
    fixed = sum((not b[q]["is_correct"]) and t[q]["is_correct"] for q in b)
    broken = sum(b[q]["is_correct"] and not t[q]["is_correct"] for q in b)
    nw = sum(not r["is_correct"] for r in b.values())
    print(f"     all sections: n={len(b)} fixed {fixed}/{nw} broken {broken}/{len(b)-nw}")
print("  split sizes:", {k: len(v) for k, v in gs["question_ids"].items()}, {k: len(v) for k, v in gs["dates"].items()})

print("\n== held-out (heldout_position.jsonl), original order / prompt note / option averaging")
rows = [json.loads(l) for l in open(R + "experiments/analysis/heldout_position.jsonl")]
H = {}
for r in rows:
    H.setdefault((r["condition"], r["set"]), {})[r["id"]] = r
for s in ("hidden", "test"):
    u, e = H[("unedited", s)], H[("editor3_edit", s)]
    ids = sorted(u)
    v = {"unedited": {q: u[q]["original_order_correct"] for q in ids},
         "prompt note": {q: u[q]["prompt_note_correct"] for q in ids},
         "averaging": {q: u[q]["option_average_correct"] for q in ids},
         "editor3": {q: e[q]["original_order_correct"] for q in ids},
         "editor3+avg": {q: e[q]["option_average_correct"] for q in ids}}
    print(f"  {s} (n={len(ids)}): " + ", ".join(f"{k} {sum(x.values())} ({100*sum(x.values())/len(ids):.1f}%)" for k, x in v.items()))
    names = list(v)
    ps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = v[names[i]], v[names[j]]
            g = sum(b[q] and not a[q] for q in ids); l = sum(a[q] and not b[q] for q in ids)
            ps.append((names[i], names[j], g, l, p_sign(g, l)))
    for x in ps:
        if "editor3+avg" not in x[:2]:
            print(f"     {x[0]} vs {x[1]}: +{x[2]} -{x[3]} p={x[4]:.3f}")
    # check original_order_correct = correct_by_position at the original answer letter
    bad = sum(u[q]["original_order_correct"] != u[q]["correct_by_position"][u[q]["answer"]] for q in ids)
    print(f"     original_order_correct consistent with correct_by_position: {bad==0}")

print("\n== other exams")
OE = R + "experiments/analysis/other_exams/"
U = json.load(open(OE + "unedited_with_choices.json"))["results"]
E = json.load(open(OE + "editor3_with_choices.json"))["results"]
RN = json.load(open(OE + "random.json"))["results"]
U1 = json.load(open(OE + "unedited.json"))["results"]
E1 = json.load(open(OE + "editor3.json"))["results"]
for t in ("belebele_sv", "belebele_en", "mmlu_math"):
    u, e = U[t], E[t]
    assert all(U1[t][q] == u[q] for q in u) and all(E1[t][q] == e[q] for q in e)
    n = len(u)
    mu = sum(sum(u[q].values()) for q in u) / (4 * n) * 100
    me = sum(sum(e[q].values()) for q in u) / (4 * n) * 100
    mr = sum(sum(RN[t][q].values()) for q in u) / (4 * n) * 100
    g = sum((not u[q]["A"]) and e[q]["A"] for q in u); l = sum(u[q]["A"] and not e[q]["A"] for q in u)
    cu, ce = U[t + "/chosen"], E[t + "/chosen"]
    # chosen = displayed letter chosen (check against correctness)
    inc = sum((cu[q][p] == p) != u[q][p] for q in u for p in "ABCD")
    pickA_u = sum(cu[q][p] == "A" for q in u for p in "ABCD") / (4 * n) * 100
    pickA_e = sum(ce[q][p] == "A" for q in u for p in "ABCD") / (4 * n) * 100
    print(f"  {t}: n={n} mean over positions unedited {mu:.2f} -> editor3 {me:.2f} (random {mr:.2f}); at A gained {g} lost {l}; "
          f"A picked {pickA_u:.1f}% -> {pickA_e:.1f}% (chosen/correct inconsistencies {inc})")

print("\n== damage (perplexity), trial_b50_k vs unedited baseline, test runs")
base = json.load(open(R + "experiments/group/gemma-4b-full/baseline/test/damage.json"))
for k in ("edited/test", "control/test"):
    d = json.load(open(R + f"experiments/editor/trial_b50_k/{k}/damage.json"))
    print(f"  {k}: ppl {base['perplexity']:.3f} -> {d['perplexity']:.3f} ({100*(d['perplexity']/base['perplexity']-1):+.2f}%); loop {base['loop_rate_percent']} -> {d['loop_rate_percent']}")

print("\n== totals")
p = json.load(open(R + "editor_runs/gemma_b50/parameters.json"))
print(f"  Gemma weights from parameters.json shapes: {sum(prod(v['shape']) for v in p.values()):,}")

print("\n== AI editors")
for b in ("gemma_b10", "gemma_b25", "gemma_b50"):
    led = [json.loads(l) for l in open(R + f"editor_runs/{b}/ledger.jsonl")]
    run = json.load(open(R + f"editor_runs/{b}/run.json"))
    fe = json.load(open(R + f"editor_runs/{b}/final_edit.json"))
    print(f"  {b}: budget {run['budget']}, passes used {led[-1]['passes_used']}, final edit weights changed {fe['size']['weights_changed']}")
print(f"  practice questions = drill {len(gs['question_ids']['drill_wrong'])+len(gs['question_ids']['drill_right'])} + validation "
      f"{len(gs['question_ids']['validation_wrong'])+len(gs['question_ids']['validation_right'])}")

print("\n== SweSAT-1.0 arithmetic (external values as given)")
for n, v in (("AI-Sweden Llama-3-8B", (71.25, 56.25, 59.69)), ("Llama-3-8B", (68.44, 65.00, 55.62)), ("Gemma-2-9b", (85.62, 82.50, 86.25))):
    print(f"  {n}: mean {sum(v)/3:.3f}")

print("\n== damage, 4-bit and text size")
q4b = json.load(open(glob.glob(R + "runs/*gemma-4b-q4_test/damage.json")[0]))
q4e = json.load(open(R + "experiments/editor/trial_b50_k/edited/q4_test/damage.json"))
print(f"  4-bit: ppl {q4b['perplexity']:.3f} -> {q4e['perplexity']:.3f} ({100*(q4e['perplexity']/q4b['perplexity']-1):+.2f}%)")
docs = [json.loads(l) for l in open("data/damage/sv_text.jsonl")]
print(f"  sv_text docs {len(docs)}, words (whitespace split) {sum(len(d['text'].split()) for d in docs):,}, "
      f"dates {min(d['published'] for d in docs)} .. {max(d['published'] for d in docs)}")
print("  baseline n_perplexity_docs:", base["n_perplexity_docs"])
