"""Independent verification of claim group G4 (group edit, gemma-4b-full). Read-only."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import hashlib, json, random
from collections import Counter
from pathlib import Path

ROOT = Path(_REPO + "")
G = Path(_RES) / "experiments/group/gemma-4b-full"
HEAD = ("LAS", "MEK")

def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def ans(cond, name): return {a["question_id"]: a for a in jl(G / cond / name / "answers.jsonl")}
def date(qid): return qid.split("/")[0]
def sec(qid): return qid.split("/")[2]
def pct(x): return f"{100*x:.2f}%"

print("=== 1. SPLIT ===")
sp = json.load(open(G / "split.json"))
qi = sp["question_ids"]
print("counts in file:", sp["counts"]); print("len of id lists:", {k: len(v) for k, v in qi.items()})
print("papers:", {k: len(v) for k, v in sp["dates"].items()})
# no paper in two groups
ds = {k: set(v) for k, v in sp["dates"].items()}
print("date overlaps:", {f"{a}&{b}": sorted(ds[a] & ds[b]) for a in ds for b in ds if a < b})
grp_dates = {g: set() for g in ds}
for k, ids in qi.items():
    grp_dates[k.split("_")[0]] |= {date(i) for i in ids}
print("dates from question ids == declared dates:", {g: grp_dates[g] == ds[g] for g in ds})
allids = [i for v in qi.values() for i in v]
print("total ids", len(allids), "unique", len(set(allids)))
# against exams/
ex = json.load(open(ROOT / "exams/split.json"))["splits"]
print("exam split keys:", {k: (len(v["exams"]), len(v["question_ids"])) for k, v in ex.items()})
dev = set(ex["dev"]["question_ids"])
print("group ids == dev ids:", set(allids) == dev, "| dev dates == union:", set(ex["dev"]["exams"]) == ds["hidden"] | ds["validation"] | ds["drill"])
for other in ("test", "gemma_extra"):
    print(f"overlap of group dates with {other}:", sorted((ds["hidden"] | ds["validation"] | ds["drill"]) & set(ex[other]["exams"])))
# reproduce the rule
order = sorted(ex["dev"]["exams"], key=lambda d: hashlib.sha256(d.encode()).hexdigest())
n = len(order); nh = round(n / 3)
print("rule reproduces hidden:", set(order[:nh]) == ds["hidden"], "validation:", set(order[nh:nh+3]) == ds["validation"], "drill:", set(order[nh+3:]) == ds["drill"], f"(n={n}, hidden={nh})")
# right/wrong from the baseline dev run
devrun = {a["question_id"]: a for a in jl(Path(_RES) / "runs/20261119T060704Z_gemma-4b-full_dev/answers.jsonl")}
mine = Counter()
bad = 0
for qid, a in devrun.items():
    g = next(g for g in ds if date(qid) in ds[g])
    key = f"{g}_{'right' if a['is_correct'] else 'wrong'}"
    mine[key] += 1
    bad += qid not in set(qi[key]) if False else 0
for k in qi:
    s = set(qi[k]); bad += sum(1 for q in s if (f"{next(g for g in ds if date(q) in ds[g])}_{'right' if devrun[q]['is_correct'] else 'wrong'}") != k)
print("my counts from the dev run:", dict(mine), "| misfiled ids:", bad)
mysec = {k: dict(Counter(sec(q) for q in v)) for k, v in qi.items()}
print("by section matches file:", all(mysec[k] == {s: c for s, c in sp["counts_by_section"][k].items() if c} for k in qi))
# passes of one paper never straddle; also check a text passage is not shared across groups
qs = {q["id"]: q for q in jl(ROOT / "exams/questions.jsonl")}
gof = {q: k.split("_")[0] for k, v in qi.items() for q in v}
ptitles = {}
for q, g in gof.items():
    t = qs[q].get("passage")
    if t: ptitles.setdefault(hashlib.md5(t.encode()).hexdigest(), set()).add(g)
print("passages appearing in >1 group:", sum(len(v) > 1 for v in ptitles.values()), "of", len(ptitles))
qtext = {}
for q, g in gof.items():
    qtext.setdefault((qs[q]["question"], tuple(sorted(qs[q]["options"].values()))), set()).add(g)
print("identical question+options in >1 group:", sum(len(v) > 1 for v in qtext.values()))

print("\n=== validation never trained on ===")
train_ids = set(qi["drill_wrong"]) | set(qi["drill_right"])
val_ids = set(qi["validation_wrong"]) | set(qi["validation_right"])
hid_ids = set(qi["hidden_wrong"]) | set(qi["hidden_right"])
print("val&train", len(val_ids & train_ids), "hid&train", len(hid_ids & train_ids), "hid&val", len(hid_ids & val_ids))
for c in ("baseline", "trained", "control"):
    for name, ids in (("drill", train_ids), ("validation", val_ids), ("hidden", hid_ids)):
        got = set(ans(c, name))
        print(f"  {c}/{name}: n={len(got)} ids match split: {got == ids}")
tr = json.load(open(G / "trained/train_result.json"))
def find(o, key, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == key: yield path + "/" + k, v
            yield from find(v, key, path + "/" + k)
for key in ("n_wrong", "n_right", "n_validation", "patience", "eval_every", "max_steps", "steps_run", "stop_reason", "min_right_kept", "lr", "shuffle"):
    print("  train_result", key, "->", sorted({json.dumps(v) for _, v in find(tr, key)}))

print("\n=== 2. TRAINING LOG ===")
log = jl(G / "trained/training_log.jsonl")
print("step  fixed kept  val_head val_all  mean_prob  is_best(log)  is_best(mine) since_best")
best = None; since = 0; stop_at = None
for r in log:
    v = r["validation"]; key = (v["headline_accuracy"], v["mean_prob_correct"])
    isb = best is None or key > best[0]
    if isb: best, since = (key, r["step"]), 0
    else: since += 1
    print(f"{r['step']:4d}  {r['n_wrong_fixed']:3d}  {r['n_right_kept']:3d}   {100*v['headline_accuracy']:.1f}    {100*v['accuracy']:.2f}   {v['mean_prob_correct']:.4f}   {r.get('is_best')!s:5}  {isb!s:5}  {since}")
    if since >= 10 and stop_at is None: stop_at = r["step"]
print("my replay: chosen step", best[1], "| patience-10 stop at step", stop_at, "| last logged step", log[-1]["step"])
print("train_result: chosen", tr["chosen_step"], "steps", tr["steps"], tr["stopping"])
print("steps at max headline:", [r["step"] for r in log if r["validation"]["headline_accuracy"] == max(x["validation"]["headline_accuracy"] for x in log)])
print("steps at max all-section val accuracy:", [(r["step"], round(r["validation"]["accuracy"], 4)) for r in log if r["validation"]["accuracy"] == max(x["validation"]["accuracy"] for x in log)])

# cross-check log rows at step 0 and at the chosen step against saved answers
bv, tv = ans("baseline", "validation"), ans("trained", "validation")
def head_acc(a): 
    h = [x for x in a.values() if x["section"] in HEAD]; return sum(x["is_correct"] for x in h), len(h)
print("baseline validation headline from answers:", head_acc(bv), "all:", sum(x["is_correct"] for x in bv.values()), "mean_prob_correct", sum(x["prob_correct"] for x in bv.values()) / len(bv))
print("trained  validation headline from answers:", head_acc(tv), "all:", sum(x["is_correct"] for x in tv.values()), "mean_prob_correct", sum(x["prob_correct"] for x in tv.values()) / len(tv))
bd, td = ans("baseline", "drill"), ans("trained", "drill")
print("trained drill from answers: fixed", sum(td[q]["is_correct"] for q in qi["drill_wrong"]), "/", len(qi["drill_wrong"]), "kept", sum(td[q]["is_correct"] for q in qi["drill_right"]), "/", len(qi["drill_right"]))

print("\n=== 3. FIXED / KEPT (full precision) ===")
def boot(after, before, samples=10000, seed=12345):
    """My own paired bootstrap (stdlib RNG, not the project's numpy code)."""
    d = [a - b for a, b in zip(after, before)]; n = len(d); rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=n)) / n for _ in range(samples))
    return sum(d) / n, means[int(0.025 * samples)], means[int(0.975 * samples) - 1]
for name in ("drill", "validation", "hidden"):
    b = ans("baseline", name)
    print(f"  baseline re-run {name} vs dev run: disagreements on is_correct", sum(b[q]["is_correct"] != devrun[q]["is_correct"] for q in b), "on letter", sum(b[q]["chosen_letter"] != devrun[q]["chosen_letter"] for q in b))
    for c in ("trained", "control"):
        a = ans(c, name)
        w, r = qi[f"{name}_wrong"], qi[f"{name}_right"]
        fx, kp = sum(a[q]["is_correct"] for q in w), sum(a[q]["is_correct"] for q in r)
        ids = sorted(a)
        d, lo, hi = boot([a[q]["is_correct"] for q in ids], [b[q]["is_correct"] for q in ids])
        print(f"  {c:8} {name:10} fixed {fx}/{len(w)} kept {kp}/{len(r)} broke {len(r)-kp} net {fx-(len(r)-kp):+d}  acc change {100*d:+.2f}pp CI [{100*lo:+.2f}, {100*hi:+.2f}] includes zero: {lo <= 0 <= hi}")
# hidden net with several seeds + exact sign test
b, a = ans("baseline", "hidden"), ans("trained", "hidden"); ids = sorted(a)
for seed in (1, 2, 3):
    d, lo, hi = boot([a[q]["is_correct"] for q in ids], [b[q]["is_correct"] for q in ids], seed=seed)
    print(f"  hidden net, seed {seed}: {100*d:+.3f}pp [{100*lo:+.3f}, {100*hi:+.3f}]")
hh = [q for q in ids if sec(q) in HEAD]
d, lo, hi = boot([a[q]["is_correct"] for q in hh], [b[q]["is_correct"] for q in hh])
print(f"  hidden LAS+MEK only (n={len(hh)}): {sum(a[q]['is_correct'] for q in hh)} vs {sum(b[q]['is_correct'] for q in hh)}  {100*d:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]")
from math import comb
def sign_p(x, y): n = x + y; k = min(x, y); return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
print("  exact sign test 6 fixed vs 4 broken: p =", round(sign_p(6, 4), 3))
# trained vs control on hidden wrong
c = ans("control", "hidden"); w = qi["hidden_wrong"]
d, lo, hi = boot([a[q]["is_correct"] for q in w], [c[q]["is_correct"] for q in w])
print(f"  hidden wrong fixed, trained - control: {100*d:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]; discordant sign test 6v1 if disjoint: p={sign_p(6,1):.3f}; same question fixed by both: {sum(a[q]['is_correct'] and c[q]['is_correct'] for q in w)}")

print("\n=== test / gemma_extra ===")
for name in ("test", "gemma_extra"):
    for cnd in ("baseline", "trained", "control"):
        x = ans(cnd, name); h = [v for v in x.values() if v["section"] in HEAD]
        bys = {s: f"{100*sum(v['is_correct'] for v in x.values() if v['section']==s)/sum(1 for v in x.values() if v['section']==s):.1f}" for s in ("ORD", "LAS", "MEK")}
        print(f"  {cnd:8} {name:11} n={len(x)} headline {sum(v['is_correct'] for v in h)}/{len(h)} = {pct(sum(v['is_correct'] for v in h)/len(h))}  all {sum(v['is_correct'] for v in x.values())}/{len(x)}  {bys}  mean prob_correct(headline) {sum(v['prob_correct'] for v in h)/len(h):.4f}")
    print("   ids match exams split:", set(ans("baseline", name)) == set(ex[name]["question_ids"]))
bt, tt, ct = ans("baseline", "test"), ans("trained", "test"), ans("control", "test")
hq = sorted(q for q in bt if sec(q) in HEAD)
for lab, x, y in (("trained vs unedited", tt, bt), ("control vs unedited", ct, bt), ("trained vs control", tt, ct)):
    d, lo, hi = boot([x[q]["prob_correct"] for q in hq], [y[q]["prob_correct"] for q in hq])
    d2, lo2, hi2 = boot([x[q]["is_correct"] for q in hq], [y[q]["is_correct"] for q in hq])
    print(f"  test headline {lab}: metric {d:+.4f} [{lo:+.4f}, {hi:+.4f}]  acc(headline) {100*d2:+.2f}pp [{100*lo2:+.2f}, {100*hi2:+.2f}]  w->r {sum(x[q]['is_correct'] and not y[q]['is_correct'] for q in bt)} r->w {sum(y[q]['is_correct'] and not x[q]['is_correct'] for q in bt)}")
stdtest = {a["question_id"]: a for a in jl(Path(_RES) / "runs/20261119T060929Z_gemma-4b-full_test/answers.jsonl")}
print("  baseline/test vs standard test run: letter disagreements", sum(bt[q]["chosen_letter"] != stdtest[q]["chosen_letter"] for q in bt))
for cnd in ("baseline", "trained", "control"):
    p = G / cnd / "test/damage.json"
    if p.exists():
        dmg = json.load(open(p)); print(f"  {cnd} damage.json:", {k: (v if not isinstance(v, (dict, list)) else "...") for k, v in dmg.items()})
for nm in ("full_trained_vs_baseline_test/test__vs__test", "full_control_vs_baseline_test/test__vs__test", "q4_trained_vs_baseline_test/q4_test__vs__20261119T061642Z_gemma-4b-q4_test"):
    cmpj = json.load(open(G / "comparisons" / nm / "comparison.json"))
    print("  comparison", nm.split("/")[0], "->", [(p, v) for p, v in find(cmpj, "verdict")][:3], [(p, v) for p, v in find(cmpj, "passed")][:1])

print("\n=== 4. 4-BIT ===")
bq, tq = ans("baseline", "q4_hidden"), ans("trained", "q4_hidden")
print("  q4 hidden ids match:", set(bq) == hid_ids, set(tq) == hid_ids)
ids = sorted(bq)
nb, nt = sum(v["is_correct"] for v in bq.values()), sum(v["is_correct"] for v in tq.values())
fx = sum(tq[q]["is_correct"] and not bq[q]["is_correct"] for q in ids); br = sum(bq[q]["is_correct"] and not tq[q]["is_correct"] for q in ids)
d, lo, hi = boot([tq[q]["is_correct"] for q in ids], [bq[q]["is_correct"] for q in ids])
print(f"  unedited q4 {nb}/480 = {pct(nb/480)}; trained q4 {nt}/480 = {pct(nt/480)}; fixed {fx}/{480-nb}, broke {br}/{nb}, net {fx-br:+d}; {100*d:+.2f}pp [{100*lo:+.2f}, {100*hi:+.2f}]")
for lab, x in (("unedited q4", bq), ("trained q4", tq)):
    h = [v for v in x.values() if v["section"] in HEAD]; print(f"  {lab} hidden LAS+MEK: {sum(v['is_correct'] for v in h)}/{len(h)} = {pct(sum(v['is_correct'] for v in h)/len(h))}")
stdq4 = {a["question_id"]: a for a in jl(Path(_RES) / "runs/20261119T061105Z_gemma-4b-q4_dev/answers.jsonl")}
print("  baseline/q4_hidden vs standard q4 dev run: letter disagreements", sum(bq[q]["chosen_letter"] != stdq4[q]["chosen_letter"] for q in bq))
fh = ans("trained", "hidden")
print("  trained full vs trained q4 hidden: full", sum(v["is_correct"] for v in fh.values()), "q4", nt)
stq = {a["question_id"]: a for a in jl(Path(_RES) / "runs/20261119T061642Z_gemma-4b-q4_test/answers.jsonl")}
ttq = ans("trained", "q4_test"); hq = sorted(q for q in ttq if sec(q) in HEAD)
d, lo, hi = boot([ttq[q]["prob_correct"] for q in hq], [stq[q]["prob_correct"] for q in hq])
print(f"  q4 test headline acc: unedited {pct(sum(stq[q]['is_correct'] for q in hq)/len(hq))} trained {pct(sum(ttq[q]['is_correct'] for q in hq)/len(hq))}; metric change {d:+.4f} [{lo:+.4f}, {hi:+.4f}]")
for cnd, name in (("baseline", "hidden"), ("trained", "hidden"), ("control", "hidden"), ("trained", "q4_hidden"), ("baseline", "q4_hidden"), ("trained", "test"), ("control", "test")):
    r = json.load(open(G / cnd / name / "run.json"))
    print(f"  run.json {cnd}/{name}: model={r.get('model')} backend={r.get('backend')} cond={r.get('experiment', {}).get('condition')} prec={r.get('experiment', {}).get('precision')}")
