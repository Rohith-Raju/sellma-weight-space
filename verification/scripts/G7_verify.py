"""Independent recomputation of group G7 claims (rotation tests on Hogskoleprovet). Read-only."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, collections
from math import comb
from pathlib import Path
ROOT = Path(_REPO + ""); G = Path(_RES); A = G / "experiments/analysis"

def sign(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]

Q = {q["id"]: q for q in jl(ROOT / "exams/questions.jsonl")}
split = json.load(open(ROOT / "exams/split.json"))["splits"]
gs = json.load(open(G / "experiments/group/gemma-4b-full/split.json"))["question_ids"]
HM = ("LAS", "MEK")
dev_ids = [i for i in split["dev"]["question_ids"] if Q[i]["section"] in HM]
print("== questions ==")
print("dev LAS+MEK:", len(dev_ids))
def dups(ids): return [i for i in ids if len(set(Q[i]["options"][L] for L in "ABCD")) < 4]
def dups_strip(ids): return [i for i in ids if len(set(Q[i]["options"][L].strip().lower() for L in "ABCD")) < 4]
print("option letter sets other than ABCD in dev LAS+MEK:", sum(sorted(Q[i]["options"]) != list("ABCD") for i in dev_ids))
print("dev duplicate option texts:", len(dups(dev_ids)), "after strip/lower:", len(dups_strip(dev_ids)))

# ---------- Claim 1
print("\n== Claim 1: rotation_proof.json ==")
R = json.load(open(A / "rotation_proof.json"))
print("conditions:", list(R))
for n, r in R.items():
    assert set(r) == set(dev_ids), n
    assert all(set(v["ok"]) == set("ABCD") for v in r.values()), n
ok = lambda n, i, L: R[n][i]["ok"][L]
for other in ("unedited_2", "unedited_after"):
    d = sum(ok("unedited_1", i, L) != ok(other, i, L) for i in dev_ids for L in "ABCD")
    md = max(abs(R["unedited_1"][i]["margin"][L] - R[other][i]["margin"][L]) for i in dev_ids for L in "ABCD")
    print(f"unedited_1 vs {other}: differing answers {d} of {4*len(dev_ids)}; max margin diff {md:.3g}")
# ok consistent with margin sign?
for n in R:
    bad = sum((R[n][i]["margin"][L] > 0) != R[n][i]["ok"][L] for i in dev_ids for L in "ABCD")
    print(f"  {n}: ok/margin-sign disagreements {bad}")
def summ(n):
    r = R[n]; N = len(dev_ids)
    pos = {L: round(100 * sum(r[i]["ok"][L] for i in dev_ids) / N, 1) for L in "ABCD"}
    cnt = {L: sum(r[i]["ok"][L] for i in dev_ids) for L in "ABCD"}
    all4 = sum(all(r[i]["ok"].values()) for i in dev_ids); none = sum(not any(r[i]["ok"].values()) for i in dev_ids)
    oaw = sum((not r[i]["ok"]["A"]) and all(r[i]["ok"][L] for L in "BCD") for i in dev_ids)
    oar = sum(r[i]["ok"]["A"] and not any(r[i]["ok"][L] for L in "BCD") for i in dev_ids)
    return pos, cnt, all4, none, N - all4 - none, oaw, oar
for n in R:
    pos, cnt, all4, none, dep, oaw, oar = summ(n)
    print(f"{n:15} pos {pos} counts {cnt} | all4 {all4} none {none} depends {dep} | wrong-only-A {oaw} right-only-A {oar}")
for n in ("editor3", "random"):
    for L in "ABCD":
        g = sum(ok(n, i, L) and not ok("unedited_1", i, L) for i in dev_ids); l = sum(ok("unedited_1", i, L) and not ok(n, i, L) for i in dev_ids)
        print(f"{n} right-at-{L}: gained {g} lost {l} p={sign(g,l):.3g}")
    a4 = lambda m, i: all(R[m][i]["ok"].values())
    g = sum(a4(n, i) and not a4("unedited_1", i) for i in dev_ids); l = sum(a4("unedited_1", i) and not a4(n, i) for i in dev_ids)
    print(f"{n} all-4: gained {g} lost {l} p={sign(g,l):.3g}")
    print(f"{n} flips vs unedited: {sum(ok(n,i,L)!=ok('unedited_1',i,L) for i in dev_ids for L in 'ABCD')}")
# other exact tests
print("wrong-only-A vs right-only-A unedited 124 vs 9 p =", sign(124, 9))
# first run
first = {r["id"]: r for r in jl(A / "rotation_first_run_dev.jsonl")}
print("first run rows", len(first), "same ids:", set(first) == set(dev_ids),
      "flips vs unedited_1:", sum(first[i][L] != ok("unedited_1", i, L) for i in dev_ids for L in "ABCD"))
# cross-check with the standard dev run (original order = shift 0, answer at the key letter)
std = {r["question_id"]: r for r in jl(G / "runs/20261119T060704Z_gemma-4b-full_dev/answers.jsonl")}
mm = [i for i in dev_ids if std[i]["is_correct"] != ok("unedited_1", i, Q[i]["answer"])]
print("standard dev run vs rotation at original key position: mismatches", len(mm), "| key matches questions.jsonl:",
      all(std[i]["correct_letter"] == Q[i]["answer"] for i in dev_ids))
# near ties
print("unedited |margin|<0.01:", sum(abs(R["unedited_1"][i]["margin"][L]) < 0.01 for i in dev_ids for L in "ABCD"))
# dev subset by group-split role (editor practised on drill papers)
role = {i: k.split("_")[0] for k, v in gs.items() for i in v}
for rl in ("drill", "validation", "hidden"):
    ids = [i for i in dev_ids if role.get(i) == rl]
    g = sum(ok("editor3", i, "A") and not ok("unedited_1", i, "A") for i in ids); l = sum(ok("unedited_1", i, "A") and not ok("editor3", i, "A") for i in ids)
    pa = lambda n: round(100 * sum(ok(n, i, "A") for i in ids) / len(ids), 1)
    print(f"dev subset {rl}: n={len(ids)} A-acc unedited {pa('unedited_1')} editor3 {pa('editor3')} gained {g} lost {l} p={sign(g,l):.3g}")
print("dev ids without a group-split role:", sum(i not in role for i in dev_ids))

# ---------- Claim 2
print("\n== Claim 2: heldout_position.jsonl ==")
H = jl(A / "heldout_position.jsonl")
by = collections.defaultdict(dict)
for r in H:
    assert r["id"] not in by[(r["condition"], r["set"])], "duplicate row"
    by[(r["condition"], r["set"])][r["id"]] = r
hidden_ids = set(gs["hidden_wrong"]) | set(gs["hidden_right"]); test_ids = set(split["test"]["question_ids"])
exp = {"hidden": {i for i in hidden_ids if Q[i]["section"] in HM}, "test": {i for i in test_ids if Q[i]["section"] in HM}}
drillval = set().union(*[set(gs[k]) for k in gs if not k.startswith("hidden")])
print("hidden exams:", sorted({i.split('/')[0] for i in hidden_ids}), "| test exams:", sorted({i.split('/')[0] for i in test_ids}))
print("hidden ∩ drill/validation:", len(hidden_ids & drillval), "| hidden ∩ test:", len(hidden_ids & test_ids), "| test ∩ any group-split id:", len(test_ids & set(role)),
      "| hidden ⊂ dev:", hidden_ids <= set(split["dev"]["question_ids"]))
# paper-level leakage: do hidden papers share exam dates/passes with drill/validation?
hp = {tuple(i.split('/')[:2]) for i in hidden_ids}; dp = {tuple(i.split('/')[:2]) for i in drillval}
print("hidden papers (date,pass):", len(hp), "overlap with drill/validation papers:", len(hp & dp),
      "| hidden exam DATES overlapping drill/val dates:", sorted({a for a, _ in hp} & {a for a, _ in dp}))
for (c, s), rows in sorted(by.items()):
    ids = set(rows)
    print(f"-- {c}/{s}: n={len(ids)} ids exactly expected set: {ids == exp[s]}; dup option texts: {len(dups(ids))} (strip/lower {len(dups_strip(ids))}); "
          f"sections {dict(collections.Counter(r['section'] for r in rows.values()))}; key matches: {all(r['answer']==Q[i]['answer'] for i,r in rows.items())}")
    rb = lambda r, L: r["correct_by_position"][L]
    N = len(rows); v = list(rows.values())
    print("   original", sum(r["original_order_correct"] for r in v), "| note", sum(r.get("prompt_note_correct", False) for r in v) if c == "unedited" else "-",
          "| averaging", sum(r["option_average_correct"] for r in v),
          "| by pos", {L: (sum(rb(r, L) for r in v), round(100 * sum(rb(r, L) for r in v) / N, 1)) for L in "ABCD"},
          "| all4", sum(all(r["correct_by_position"].values()) for r in v), "none", sum(not any(r["correct_by_position"].values()) for r in v),
          "| wrong-only-A", sum((not rb(r, "A")) and all(rb(r, L) for L in "BCD") for r in v),
          "right-only-A", sum(rb(r, "A") and not any(rb(r, L) for L in "BCD") for r in v))
    # internal consistency: shift 0 is the original order
    print("   original_order_correct == correct_by_position[key]: mismatches", sum(r["original_order_correct"] != rb(r, r["answer"]) for r in v))
    # logical bounds on averaging: right in all 4 => averaging right is NOT implied, but report
    print("   averaging right & none4:", sum(r["option_average_correct"] and not any(r["correct_by_position"].values()) for r in v),
          "| averaging wrong & all4:", sum((not r["option_average_correct"]) and all(r["correct_by_position"].values()) for r in v))
for s in ("hidden", "test"):
    u, e = by[("unedited", s)], by[("editor3_edit", s)]
    for L in "ABCD":
        g = sum(e[i]["correct_by_position"][L] and not u[i]["correct_by_position"][L] for i in u); l = sum(u[i]["correct_by_position"][L] and not e[i]["correct_by_position"][L] for i in u)
        print(f"{s} right-at-{L} gained {g} lost {l} p={sign(g,l):.3g}")
    for name, key, a, b in (("orig: editor3 vs unedited", "original_order_correct", u, e), ("averaging vs original (unedited)", None, u, u), ("note vs original", "note", u, u)):
        if key == "original_order_correct": f1 = lambda i: u[i][key]; f2 = lambda i: e[i][key]
        elif key is None: f1 = lambda i: u[i]["original_order_correct"]; f2 = lambda i: u[i]["option_average_correct"]
        else: f1 = lambda i: u[i]["original_order_correct"]; f2 = lambda i: u[i]["prompt_note_correct"]
        g = sum(f2(i) and not f1(i) for i in u); l = sum(f1(i) and not f2(i) for i in u)
        print(f"{s} {name}: gained {g} lost {l} p={sign(g,l):.3g}")
# cross-check original-order results with the independent standard runs
src = {("unedited", "hidden"): G / "experiments/group/gemma-4b-full/baseline/hidden/answers.jsonl",
       ("unedited", "test"): G / "runs/20261119T060929Z_gemma-4b-full_test/answers.jsonl",
       ("editor3_edit", "hidden"): G / "experiments/editor/trial_b50_k/edited/hidden/answers.jsonl",
       ("editor3_edit", "test"): G / "experiments/editor/trial_b50_k/edited/test/answers.jsonl"}
for k, p in src.items():
    s = {r["question_id"]: r for r in jl(p)}
    rows = by[k]
    print(f"cross-check {k} vs {p.relative_to(G)}: missing {sum(i not in s for i in rows)}, original-order mismatches {sum(s[i]['is_correct'] != rows[i]['original_order_correct'] for i in rows if i in s)}, standard-run LAS+MEK right {sum(s[i]['is_correct'] for i in rows if i in s)}")
# summary json agrees with jsonl?
S = json.load(open(A / "heldout_position.json"))["summary"]
print("summary json:", json.dumps(S))
