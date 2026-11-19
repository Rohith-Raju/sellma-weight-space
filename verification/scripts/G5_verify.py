"""G5 verifier: AI editor runs. Read-only. Run: uv run --extra gpu python -I verification/scripts/G5_verify.py"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json, os, hashlib
from collections import Counter
from pathlib import Path

ROOT = Path(_REPO + "")
ER = Path(_RES) / "editor_runs"
BASE = Path(_RES) / "experiments/group/gemma-4b-full/baseline"

def jl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

Q = {q["id"]: q for q in jl(ROOT / "exams/questions.jsonl")}
base = {}
for part in ("drill", "validation"):
    for a in jl(BASE / part / "answers.jsonl"):
        assert a["question_id"] not in base
        base[a["question_id"]] = dict(a, paper=part)
print("practice set:", len(base), Counter(a["paper"] for a in base.values()))
# baseline answers.jsonl keys vs questions.jsonl keys
assert all(Q[i]["answer"] == a["correct_letter"] for i, a in base.items())
split = json.load(open(Path(_RES) / "experiments/group/gemma-4b-full/split.json"))
practice_dates = set(split["dates"]["drill"]) | set(split["dates"]["validation"])

def choose(q, probs):
    letters = sorted(q["options"])
    lp = {l: float(probs.get(l, 0.0)) for l in letters}
    return max(letters, key=lambda l: lp[l])

def shown_key(q, order):
    letters = sorted(q["options"])
    if order is None: return q["answer"]
    assert sorted(order) == letters
    return letters[list(order).index(q["answer"])]

print("\n=== 1. ledgers / passes used ===")
for run, budget in (("gemma_b10", 10), ("gemma_b25", 25), ("gemma_b50", 50)):
    led = jl(ER / run / "ledger.jsonl")
    rj = json.load(open(ER / run / "run.json")); fe = json.load(open(ER / run / "final_edit.json"))
    cost = sum(e.get("cost", 0) for e in led)
    kinds = [e["kind"] for e in led]
    folders = sorted(os.listdir(ER / run / "passes"))
    led_folders = [e["folder"].split("/")[1] for e in led if e.get("folder")]
    mono = all(led[i]["time"] <= led[i + 1]["time"] for i in range(len(led) - 1))
    seq = [e["passes_used"] for e in led if e["kind"] != "submit"]
    print(run, "budget", rj["budget"], "ledger cost", cost, "final_edit.passes_used", fe["passes_used"],
          "kinds", Counter(kinds), "| folders==ledger:", folders == led_folders, "| times monotone:", mono,
          "| passes_used sequence ok:", seq == list(range(1, len(seq) + 1)),
          "| every cost-1 entry has folder:", all(("folder" in e) == (e["kind"] != "submit") for e in led),
          "| submit last & once:", kinds.count("submit") == 1 and kinds[-1] == "submit",
          "| created", rj["created_at"], "first", led[0]["time"], "submit", led[-1]["time"])
    # n_items vs request, and file mtimes not available reliably (copied); check request counts
    for e in led:
        if e.get("folder"):
            rq = json.load(open(ER / run / e["folder"] / "request.json"))
            assert rq["kind"] == e["kind"] and len(rq["texts"]) == e["n_items"], (run, e)
            has_edit = (ER / run / e["folder"] / "edit.safetensors").exists()
            assert has_edit == (e["edit"] is not None), (run, e["folder"])

print("\n=== 1b/5. edit files ===")
import torch
from safetensors import safe_open
from safetensors.torch import load_file
def load_edit(p):
    with safe_open(str(p), framework="pt") as f: forms = json.loads(f.metadata()["forms"])
    return forms, load_file(str(p))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
def describe(p):
    forms, t = load_edit(p)
    out = {}
    for name, form in forms.items():
        v = t[f"values:{name}"]
        idx = t.get(f"{form}:{name}")
        out[name.replace("model.language_model.layers.", "L")] = dict(
            form=form, shape=tuple(v.shape), nonzero=int(torch.count_nonzero(v)), absmax=float(v.abs().max()),
            norm=round(float(torch.linalg.vector_norm(v.float())), 6),
            idx=(None if idx is None else (int(idx.min()), int(idx.max()), len(idx))),
            finite=bool(torch.isfinite(v).all()))
    return out
for run in ("gemma_b10", "gemma_b25", "gemma_b50", "trial_b10_q_full", "trial_b10_q_half", "trial_b25_mlp", "trial_b50_k"):
    d = describe(ER / run / "final_edit.safetensors")
    fe = json.load(open(ER / run / "final_edit.json"))
    tot = sum(x["nonzero"] for x in d.values())
    print(run, "sha", sha(ER / run / "final_edit.safetensors"), "| nonzero entries in file:", tot,
          "| json touched/changed:", fe["size"]["weights_touched"], fe["size"]["weights_changed"],
          "| share %.5f%%" % (100 * fe["size"]["share_of_model"]))
    for k, v in d.items(): print("    ", k, v)
    led = jl(ER / run / "ledger.jsonl"); print("     ledger:", [(e["kind"], e["cost"]) for e in led])
print("trial edit == which editor pass edit (byte-identical tensors)?")
for trial, (run, p) in {"trial_b10_q_full": ("gemma_b10", "003_answer"), "trial_b10_q_half": ("gemma_b10", "004_answer"),
                        "trial_b25_mlp": ("gemma_b25", "006_answer"), "trial_b50_k": ("gemma_b50", "009_answer")}.items():
    f1, t1 = load_edit(ER / trial / "final_edit.safetensors"); f2, t2 = load_edit(ER / run / "passes" / p / "edit.safetensors")
    same = f1 == f2 and t1.keys() == t2.keys() and all(torch.equal(t1[k], t2[k]) for k in t1)
    print("  ", trial, "==", run, p, ":", same)
    for k in t1:
        if k.startswith(("rows:", "cols:")): 
            ix = sorted(t1[k].tolist()); print("      ", k.split("layers.")[1], "index range", ix[0], ix[-1], "n", len(ix), "contiguous blocks:", [ (a,b) for a,b in _blocks(ix)] if False else "")
print("b10 pass5 edit == pass4 edit:", all(torch.equal(a, b) for a, b in zip(load_edit(ER/"gemma_b10/passes/005_answer/edit.safetensors")[1].values(), load_edit(ER/"gemma_b10/passes/004_answer/edit.safetensors")[1].values())))
def blocks(ix):
    out=[]; s=ix[0]; p=ix[0]
    for i in ix[1:]:
        if i!=p+1: out.append((s,p)); s=i
        p=i
    out.append((s,p)); return out
for trial in ("trial_b10_q_full","trial_b50_k"):
    f,t=load_edit(ER/trial/"final_edit.safetensors")
    for k in t:
        if k.startswith(("rows:","cols:")): print("  ",trial,k.split("layers.")[1],"blocks",blocks(sorted(t[k].tolist())))
print("  share check: 655360/4300079472 = %.5f%%; 366231/4300079472 = %.5f%%" % (100*655360/4300079472, 100*366231/4300079472))

print("\n=== 3. answer passes rescored from result.json ===")
def score_pass(run, p):
    rq = json.load(open(ER / run / "passes" / p / "request.json")); rs = json.load(open(ER / run / "passes" / p / "result.json"))
    assert len(rq["question_ids"]) == len(rs["letter_probs"]) == len(rq["options_orders"])
    rows = []
    for qid, order, probs in zip(rq["question_ids"], rq["options_orders"], rs["letter_probs"]):
        assert qid in base, ("NOT PRACTICE", qid)
        q = Q[qid]; key = shown_key(q, order); ch = choose(q, probs)
        b = base[qid]
        rows.append(dict(id=qid, sec=q["section"], order=order, key=key, ch=ch, ok=ch == key,
                         b_ok=b["is_correct"], b_ch=b["chosen_letter"], b_key=b["correct_letter"]))
    return rows
def summarize(run, p, rows, note=""):
    n = len(rows); after = sum(r["ok"] for r in rows); before = sum(r["b_ok"] for r in rows)
    fixed = sum(r["ok"] and not r["b_ok"] for r in rows); broke = sum(r["b_ok"] and not r["ok"] for r in rows)
    secs = Counter(r["sec"] for r in rows); dup = n - len({(r["id"], r["order"]) for r in rows})
    persec = {s: (sum(r["b_ok"] for r in rows if r["sec"] == s), sum(r["ok"] for r in rows if r["sec"] == s)) for s in secs}
    print(f"{run} {p}: n={n} {dict(secs)} dup={dup} permuted={sum(r['order'] is not None for r in rows)} | "
          f"baseline(orig order) {before} -> edited {after} | fixed {fixed} broke {broke} | per section before->after {persec} {note}")
    return before, after, fixed, broke
allrows = {}
for run in ("gemma_b10", "gemma_b25", "gemma_b50"):
    for p in sorted(os.listdir(ER / run / "passes")):
        if p.endswith("answer"):
            rows = score_pass(run, p); allrows[(run, p)] = rows; summarize(run, p, rows)
            four = [r for r in rows if r["sec"] != "ORD" and r["order"] is None]
            if four:
                print("      4-option, original order: chosen", sorted(Counter(r["ch"] for r in four).items()), "baseline chosen", sorted(Counter(r["b_ch"] for r in four).items()),
                      "| keyA right", sum(r["ok"] for r in four if r["key"] == "A"), "of", sum(r["key"] == "A" for r in four), "(base", sum(r["b_ok"] for r in four if r["key"] == "A"), ")")
# b50 whole practice: pass 9 (T1) + pass 10 (T2)
r9, r10 = allrows[("gemma_b50", "009_answer")], allrows[("gemma_b50", "010_answer")]
ids = [r["id"] for r in r9 + r10]
print("b50 pass9+pass10: distinct ids", len(set(ids)), "of", len(ids), "covers whole practice set:", set(ids) == set(base))
summarize("gemma_b50", "009+010", r9 + r10)
mekA = [r for r in r9 + r10 if r["sec"] == "MEK" and r["key"] == "A"]
print("   MEK key=A:", sum(r["b_ok"] for r in mekA), "->", sum(r["ok"] for r in mekA), "of", len(mekA))
# b10 pass 5: permuted. baseline in reversed order is not in the unedited answers; report vs original-order baseline per part
r5 = allrows[("gemma_b10", "005_answer")]
for nm, sel in (("ORD orig", lambda r: r["sec"] == "ORD"), ("MEK DCBA", lambda r: r["sec"] == "MEK"), ("LAS DCBA", lambda r: r["sec"] == "LAS")):
    s = [r for r in r5 if sel(r)]
    print("   b10 pass5", nm, "n", len(s), "orders", Counter(r["order"] for r in s), "edited correct", sum(r["ok"] for r in s), "| original-order baseline correct", sum(r["b_ok"] for r in s))

print("\n=== 4. diagnoses from the unedited practice answers ===")
P4 = [a for a in base.values() if a["section"] in ("MEK", "LAS")]
print("4-option practice questions:", len(P4), "| A chosen", sum(a["chosen_letter"] == "A" for a in P4), "| key A", sum(a["correct_letter"] == "A" for a in P4))
for sec in ("ORD", "LAS", "MEK"):
    s = [a for a in base.values() if a["section"] == sec]
    acc = sum(a["is_correct"] for a in s) / len(s)
    bykey = {k: (sum(a["is_correct"] for a in s if a["correct_letter"] == k), sum(a["correct_letter"] == k for a in s)) for k in "ABCDE" if any(a["correct_letter"] == k for a in s)}
    print(sec, "n", len(s), "correct", sum(a["is_correct"] for a in s), "acc %.3f" % acc, "| chosen", sorted(Counter(a["chosen_letter"] for a in s).items()),
          "| key", sorted(Counter(a["correct_letter"] for a in s).items()), "| acc by key", {k: "%d/%d=%.1f%%" % (r, n, 100 * r / n) for k, (r, n) in bykey.items()})
print("total correct", sum(a["is_correct"] for a in base.values()), "of", len(base), "wrong", sum(not a["is_correct"] for a in base.values()))
# what subset gives 86?
for nm, sel in (("drill only", lambda a: a["paper"] == "drill"), ("validation only", lambda a: a["paper"] == "validation")):
    s = [a for a in P4 if sel(a)]; print("  ", nm, "n", len(s), "A chosen", sum(a["chosen_letter"] == "A" for a in s), "key A", sum(a["correct_letter"] == "A" for a in s))
for run, p in (("gemma_b10", "003_answer"), ("gemma_b25", "003_answer")):
    s = allrows[(run, p)]; print("  ", run, p, "set (", len(s), "): baseline A chosen", sum(r["b_ch"] == "A" for r in s), "key A", sum(r["b_key"] == "A" for r in s), "baseline correct", sum(r["b_ok"] for r in s))

print("\n=== 4b. b50 rotation diagnosis from pass 1 read (top_tokens order; probabilities file not in the Mac copy) ===")
rq = json.load(open(ER / "gemma_b50/passes/001_read/request.json")); rs = json.load(open(ER / "gemma_b50/passes/001_read/result.json"))
meta = json.load(open(ER / "gemma_b50/notes/pass1_meta.json"))
print("texts", len(rq["texts"]), "meta", len(meta), "result.safetensors present:", (ER / "gemma_b50/passes/001_read/result.safetensors").exists())
# independent mapping of text -> (question, order): find the practice question whose options appear in the text, derive order from text
import re
def parse_prompt(text):
    opts = re.findall(r"\n([A-E])\. (.*)", text.split("Svara endast")[0])
    return [o for _, o in opts[-4:]]
qtext_index = {}
for qid in base:
    q = Q[qid]
    if q["section"] in ("MEK", "LAS"): qtext_index.setdefault(tuple(sorted(q["options"].values())), []).append(qid)
def first_letter(tokens):
    for t in tokens:
        s = t.strip()
        if s in ("A", "B", "C", "D"): return s
    return None
ch = {}; mism = 0; nolet = 0
for i, ((qid, order), text) in enumerate(zip(meta, rq["texts"])):
    q = Q[qid]; shown = parse_prompt(text)
    exp = [q["options"][l] for l in (order or "ABCD")]
    if shown != exp or (q["question"] not in text): mism += 1
    c = first_letter(rs["top_tokens"][i])
    if c is None: nolet += 1
    ch[(qid, order)] = c
print("meta/order vs prompt text mismatches:", mism, "| prompts with no letter in top-k:", nolet)
agree = sum(ch[(qid, None)] == base[qid]["chosen_letter"] for qid, o in meta if o is None); n0 = sum(o is None for _, o in meta)
print("original-order top-token choice agrees with stored baseline answers:", agree, "of", n0)
for sec in ("MEK", "LAS"):
    tabk = {}; same = 0; n = 0
    for (qid, o) in ch:
        if o is None or Q[qid]["section"] != sec: continue
        q = Q[qid]; k0 = q["answer"]; k1 = shown_key(q, o)
        a = ch[(qid, None)] == k0; b = ch[(qid, o)] == k1
        tabk.setdefault(k0 + "->" + k1, Counter())[("right" if a else "wrong") + ">" + ("right" if b else "wrong")] += 1
        n += 1; same += ch[(qid, None)] == o["ABCD".index(ch[(qid, o)])] if ch[(qid, o)] else 0
    print(sec, "rotated pairs", n, "same content chosen", same, "(%.1f%% change)" % (100 * (1 - same / n)))
    for k, v in sorted(tabk.items()): print("    key", k, dict(v), "| fixed", v["wrong>right"], "broke", v["right>wrong"])
# multi-gap first-position accuracy
for rot in (None, "BCDA"):
    for multi in (True, False):
        s = [(qid, o) for (qid, o) in ch if o == rot and Q[qid]["section"] == "MEK" and ((" – " in Q[qid]["options"]["A"]) == multi)]
        acc = {}
        for l in "ABCD":
            ss = [x for x in s if shown_key(Q[x[0]], x[1]) == l]
            acc[l] = "%d/%d=%.0f%%" % (sum(ch[x] == l for x in ss), len(ss), 100 * sum(ch[x] == l for x in ss) / max(1, len(ss)))
        print("  MEK", "rot" if rot else "orig", "multi-gap" if multi else "single", acc)

print("\n=== 4c. b10/b25 reversed-order figures from their pass 1 reads (top_tokens) ===")
for run in ("gemma_b10", "gemma_b25"):
    rq = json.load(open(ER / run / "passes/001_read/request.json")); rs = json.load(open(ER / run / "passes/001_read/result.json"))
    # identify each text: question by sentence/options, order by option sequence
    byopts = {}
    for qid in base:
        q = Q[qid]
        if q["section"] in ("MEK", "LAS"): byopts.setdefault((q["section"], tuple(q["options"][l] for l in "ABCD")), []).append(qid)
    rec = {}; unk = 0
    for text, toks in zip(rq["texts"], rs["top_tokens"]):
        shown = tuple(parse_prompt(text)); found = None
        for order in (None, "DCBA"):
            orig = shown if order is None else tuple(reversed(shown))
            for sec in ("MEK", "LAS"):
                for qid in byopts.get((sec, orig), []):
                    q = Q[qid]
                    if q["question"] in text and (not q.get("passage") or q["passage"][:200] in text): found = (qid, order)
            if found: break
        if not found: unk += 1; continue
        rec[found] = first_letter(toks)
    print(run, "texts", len(rq["texts"]), "identified", len(rec), "unidentified", unk, "orders", Counter(o for _, o in rec))
    ag = [(rec[k] == base[k[0]]["chosen_letter"]) for k in rec if k[1] is None]; print("   original-order agreement with stored baseline:", sum(ag), "of", len(ag))
    for sec in ("MEK", "LAS"):
        rv = [k for k in rec if k[1] == "DCBA" and Q[k[0]]["section"] == sec]
        kA = [k for k in rv if shown_key(Q[k[0]], "DCBA") == "A"]
        same = sum(rec.get((qid, None)) == "DCBA"["ABCD".index(rec[(qid, o)])] for qid, o in rv if rec[(qid, o)] and (qid, None) in rec)
        print("   ", sec, "reversed n", len(rv), "| shown-key-A right %d/%d=%.1f%%" % (sum(rec[k] == "A" for k in kA), len(kA), 100 * sum(rec[k] == "A" for k in kA) / max(1, len(kA))),
              "| same content both orders %d/%d=%.1f%%" % (same, len(rv), 100 * same / max(1, len(rv))),
              "| reversed correct", sum(rec[k] == shown_key(Q[k[0]], "DCBA") for k in rv))
    if run == "gemma_b10":
        # pass 5 reversed baseline: 375 -> 374 claim
        r5 = allrows[("gemma_b10", "005_answer")]; b = 0; miss = 0
        for r in r5:
            if r["order"] is None: b += r["b_ok"]
            elif (r["id"], r["order"]) in rec: b += rec[(r["id"], r["order"])] == r["key"]
            else: miss += 1
        print("   b10 pass5 baseline (ORD stored baseline + reversed from pass-1 read):", b, "missing", miss, "-> edited", sum(r["ok"] for r in r5))

print("\n=== every question any pass touched is a practice question (dates) ===")
for run in ("gemma_b10", "gemma_b25", "gemma_b50"):
    bad = 0; n = 0
    for p in sorted(os.listdir(ER / run / "passes")):
        rq = json.load(open(ER / run / "passes" / p / "request.json"))
        if rq["kind"] == "answer":
            for qid in rq["question_ids"]:
                n += 1; bad += qid.split("/")[0] not in practice_dates
    print(run, "answer-pass question ids:", n, "outside practice dates:", bad)
