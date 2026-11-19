"""G6 verifier: trial edits at full precision and 4-bit. Read-only; recomputes from answers.jsonl."""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, re, random, hashlib, os
from collections import Counter
from pathlib import Path

ROOT = Path(_REPO + "")
G = Path(_RES)
E = G / "experiments/editor"
B = G / "experiments/group/gemma-4b-full/baseline"
Q4TEST = G / "runs/20261119T061642Z_gemma-4b-q4_test"
UNEDITED_Q4 = "fe8598adf4f20d50d88bf8877540013717051e869faff95117de56753a8cd127"
TRIALS = ["trial_b50_k", "trial_b10_q_full", "trial_b25_mlp", "trial_b10_q_half"]

keys = {}
for line in open(ROOT / "exams/questions.jsonl"):
    q = json.loads(line); keys[q["id"]] = (q["section"], q["answer"])
gsplit = json.load(open(G / "experiments/group/gemma-4b-full/split.json"))
hidden_ids = set(gsplit["question_ids"]["hidden_wrong"]) | set(gsplit["question_ids"]["hidden_right"])
test_ids = set(json.load(open(ROOT / "exams/split.json"))["splits"]["test"]["question_ids"])

problems = []
def load(d):
    rows = {}
    for line in open(Path(d) / "answers.jsonl"):
        r = json.loads(line)
        qid = r["question_id"]
        assert qid not in rows, f"dup {qid} in {d}"
        sec, ans = keys[qid]
        if r["section"] != sec or r["correct_letter"] != ans:
            problems.append(f"{d}: key/section mismatch {qid}")
        lp = r["letter_probs"]; arg = max(lp, key=lp.get)
        if arg != r["chosen_letter"]:
            problems.append(f"{d}: chosen != argmax {qid}")
        if (r["chosen_letter"] == ans) != r["is_correct"]:
            problems.append(f"{d}: is_correct inconsistent {qid}")
        rows[qid] = r
    return rows

def stats(rows):
    lm = [r for q, r in rows.items() if keys[q][0] in ("LAS", "MEK")]
    n = len(lm); ka = [r for r in lm if keys[r["question_id"]][1] == "A"]
    kb = [r for r in lm if keys[r["question_id"]][1] != "A"]
    corr = lambda rs: sum(r["chosen_letter"] == keys[r["question_id"]][1] for r in rs)
    return dict(n=n, keyA=len(ka), all=100*corr(lm)/n, Aright=100*corr(ka)/len(ka),
                BCD=100*corr(kb)/len(kb), Apicked=sum(r["chosen_letter"] == "A" for r in lm),
                c_all=corr(lm), c_A=corr(ka), c_BCD=corr(kb))

runs = {
    "unedited full hidden": B / "hidden", "unedited q4 hidden": B / "q4_hidden",
    "unedited full test": B / "test", "unedited q4 test": Q4TEST,
}
for t in TRIALS:
    runs[f"{t} full hidden"] = E / t / "edited/hidden"
    runs[f"{t} q4 hidden"] = E / t / "edited/q4_hidden"
    runs[f"{t} control hidden"] = E / t / "control/hidden"
    runs[f"{t} full test"] = E / t / "edited/test"
    runs[f"{t} q4 test"] = E / t / "edited/q4_test"
    runs[f"{t} control test"] = E / t / "control/test"
data = {k: load(v) for k, v in runs.items()}

print("== id coverage")
for k, rows in data.items():
    ref = hidden_ids if "hidden" in k else test_ids
    ok = set(rows) == ref
    print(f"{k:32s} n={len(rows)} same-as-reference-ids={ok}")
    if not ok: problems.append(f"{k}: id set differs from reference")

print("\n== LAS+MEK stats  (n, keyA, all%, A-right%, BCD%, A picked | counts)")
S = {}
for k, rows in data.items():
    s = S[k] = stats(rows)
    print(f"{k:32s} {s['n']} {s['keyA']} {s['all']:.1f} / {s['Aright']:.1f} / {s['BCD']:.1f} / {s['Apicked']}"
          f"   | {s['c_all']}/{s['n']}, {s['c_A']}/{s['keyA']}, {s['c_BCD']}/{s['n']-s['keyA']}")

print("\n== A-right gains for trial_b50_k (unrounded)")
print("full hidden:", S["trial_b50_k full hidden"]["Aright"] - S["unedited full hidden"]["Aright"])
print("q4 hidden:  ", S["trial_b50_k q4 hidden"]["Aright"] - S["unedited q4 hidden"]["Aright"])
print("full test:  ", S["unedited full test"]["Aright"], "->", S["trial_b50_k full test"]["Aright"])

def paired(base, ed, secs, seed=0, B_=10000):
    ids = sorted(q for q in base if keys[q][0] in secs)
    b = [base[q]["is_correct"] for q in ids]; e = [ed[q]["is_correct"] for q in ids]
    fixed = sum((not x) and y for x, y in zip(b, e)); broken = sum(x and (not y) for x, y in zip(b, e))
    d = [int(y) - int(x) for x, y in zip(b, e)]; n = len(d)
    rng = random.Random(seed); bs = []
    for _ in range(B_):
        bs.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    bs.sort()
    return dict(n=n, fixed=fixed, broken=broken, net=fixed - broken, pp=100*sum(d)/n,
                lo=100*bs[int(0.025*B_)], hi=100*bs[int(0.975*B_)-1])

print("\n== 4-bit hidden net, trial_b50_k vs unedited 4-bit (paired bootstrap, 10k)")
for secs, lab in [(("ORD", "LAS", "MEK"), "all 480"), (("LAS", "MEK"), "LAS+MEK")]:
    print(lab, paired(data["unedited q4 hidden"], data["trial_b50_k q4 hidden"], secs))
print("full hidden, b50_k vs unedited full:", paired(data["unedited full hidden"], data["trial_b50_k full hidden"], ("ORD","LAS","MEK")))

# ---- damage
def loops(reply):
    w = re.sub(r"[^\w\s]", " ", reply.lower()).split()
    c = Counter(tuple(w[i:i+4]) for i in range(len(w) - 3))
    return any(v >= 3 for v in c.values())
def loop_rate(d):
    p = Path(d) / "generations.jsonl"
    g = [json.loads(l) for l in open(p)]
    return 100 * sum(loops(x["reply"]) for x in g) / len(g), len(g), [x["id"] for x in g]

print("\n== damage (test split)")
dec = dict(ppl=0.05, loop=2.0)
refs = {"full": B / "test", "q4": Q4TEST}
base_dmg = {}
for p, d in refs.items():
    dj = json.load(open(d / "damage.json")); lr, ng, gids = loop_rate(d)
    base_dmg[p] = (dj, lr, gids)
    print(f"unedited {p}: ppl {dj['perplexity']:.4f} loop(damage.json) {dj['loop_rate_percent']} loop(recomputed) {lr:.1f} on {ng}")
for t in TRIALS:
    for cond, sub, p in [("edited", "test", "full"), ("control", "test", "full"), ("edited", "q4_test", "q4")]:
        d = E / t / cond / sub
        dj = json.load(open(d / "damage.json")); lr, ng, gids = loop_rate(d)
        bdj, blr, bgids = base_dmg[p]
        if gids != bgids: problems.append(f"{d}: generation prompt ids differ from baseline")
        rel = (dj["perplexity"] - bdj["perplexity"]) / bdj["perplexity"]
        pts = dj["loop_rate_percent"] - bdj["loop_rate_percent"]
        within = rel <= dec["ppl"] and pts <= dec["loop"]
        print(f"{t:18s} {cond:7s} {p:4s} ppl {dj['perplexity']:.4f} ({rel:+.2%}) loop {dj['loop_rate_percent']} "
              f"(recomputed {lr:.1f}) change {pts:+.1f} pts -> {'within limits' if within else 'FLAGGED'}")

print("\n== 4-bit GGUF identity")
for t in TRIALS:
    for sub in ["q4_hidden", "q4_test", "q4_gemma_extra"]:
        rj = json.load(open(E / t / "edited" / sub / "run.json"))
        ds = rj["describe"]
        print(f"{t:18s} {sub:15s} sha={ds['sha256'][:12]} differs_from_unedited={ds['sha256'] != UNEDITED_Q4} "
              f"path={ds['path']} model={rj['model']} edit_run={rj['experiment'].get('editor_run')}")
        if ds["sha256"] == UNEDITED_Q4: problems.append(f"{t} {sub}: used unedited GGUF")
for lab, d in [("baseline q4_hidden", B / "q4_hidden"), ("q4 test run", Q4TEST)]:
    print(lab, json.load(open(d / "run.json"))["describe"]["sha256"][:12])

print("\n== full-precision runs: in-memory edit metadata")
for t in TRIALS:
    for cond in ["edited", "control"]:
        rj = json.load(open(E / t / cond / "hidden/run.json"))
        ac = rj["experiment"].get("applied_change", [])
        print(t, cond, rj["model"], "entries_changed", sum(a["entries_changed"] for a in ac),
              "tensors", len(ac))

print("\n== identical-answer check (edited vs unedited, full hidden)")
for t in TRIALS:
    a = data[f"{t} full hidden"]; b = data["unedited full hidden"]
    nd = sum(a[q]["chosen_letter"] != b[q]["chosen_letter"] for q in a)
    dp = sum(abs(a[q]["prob_correct"] - b[q]["prob_correct"]) > 1e-6 for q in a)
    print(t, "chosen differs on", nd, "prob_correct differs on", dp)

print("\n== problems")
for p in problems: print(" -", p)
print("none" if not problems else f"{len(problems)} problems")
