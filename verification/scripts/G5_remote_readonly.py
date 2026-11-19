"""G5 remote check (GPU machine, CPU only, read-only; prints only).
Run from the Mac:  ssh sellma-gpu 'cd /root/sellma && .venv/bin/python -I -' < verification/scripts/G5_remote_readonly.py
(a) weights really changed by each final/trial edit, recomputed against the stored bfloat16 weights
(b) b50 pass-1 rotation table from the stored top-k probabilities
(c) sha256 of ledgers, final edits, scripts so the Mac copy can be compared; file listing of run folders."""
import json, hashlib, os, torch
from pathlib import Path
from collections import Counter
from safetensors import safe_open
from safetensors.torch import load_file
torch.set_grad_enabled(False)
R = Path("/root/sellma"); ER = R / "editor_runs"; MD = R / "models/gemma-3-4b-it-official"

print("== (a) weights changed")
for run in ("gemma_b10", "gemma_b25", "gemma_b50", "trial_b10_q_full", "trial_b10_q_half", "trial_b25_mlp", "trial_b50_k"):
    table = json.load(open(ER / run / "parameters.json"))
    p = ER / run / "final_edit.safetensors"
    with safe_open(str(p), framework="pt") as f: forms = json.loads(f.metadata()["forms"])
    t = load_file(str(p)); tot_changed = tot_touched = 0
    for name, form in forms.items():
        e = table[name]
        with safe_open(str(MD / e["file"]), framework="pt") as f: W = f.get_tensor(e["key"])
        assert str(W.dtype) == "torch.bfloat16", W.dtype
        v = t["values:" + name]
        sel = (Ellipsis,) if form == "dense" else (t["rows:" + name],) if form == "rows" else (slice(None), t["cols:" + name])
        orig = W[sel]; new = (orig.to(torch.float32) + v.to(torch.float32)).to(W.dtype)
        ch = int((new != orig).sum()); tot_changed += ch; tot_touched += int(torch.count_nonzero(v))
    total = sum(int(torch.tensor(e["shape"]).prod()) for e in table.values())
    print(run, "touched", tot_touched, "changed", tot_changed, "model weights (parameter table)", total, "share touched %.5f%%" % (100 * tot_touched / total))

print("== (b) b50 pass 1 rotation table from top-k probabilities")
run = ER / "gemma_b50"
rq = json.load(open(run / "passes/001_read/request.json")); meta = json.load(open(run / "notes/pass1_meta.json"))
with safe_open(str(run / "passes/001_read/result.safetensors"), framework="pt") as f:
    ids = f.get_tensor("top_ids"); pr = f.get_tensor("top_probs")
lt = json.load(open(run / "letter_token_ids.json")); lt2 = json.load(open(run / "notes/letter_tokens.json"))
print("letter sets: workbench", {k: len(v) for k, v in lt.items()}, "editor", {k: len(v) for k, v in lt2.items()})
Q = {}
for l in open(R / "exams/questions.jsonl"):
    q = json.loads(l); Q[q["id"]] = q
def table(sets):
    ch = {}
    for i, (qid, o) in enumerate(meta):
        d = {l: 0.0 for l in "ABCD"}
        for tkn, p in zip(ids[i].tolist(), pr[i].tolist()):
            for l in "ABCD":
                if tkn in sets[l]: d[l] += p
        ch[(qid, o)] = max("ABCD", key=lambda l: d[l])
    for sec in ("MEK", "LAS"):
        tab = {}; same = n = 0
        for (qid, o) in ch:
            if o is None or Q[qid]["section"] != sec: continue
            k0 = Q[qid]["answer"]; k1 = "ABCD"[o.index(k0)]
            a = ch[(qid, None)] == k0; b = ch[(qid, o)] == k1
            tab.setdefault(k0 + "->" + k1, Counter())[("fixed" if b else "wrong") if not a else ("kept" if b else "broke")] += 1
            n += 1; same += ch[(qid, None)] == o["ABCD".index(ch[(qid, o)])]
        print("  ", sec, "pairs", n, "content changed %.1f%%" % (100 * (1 - same / n)), {k: dict(v) for k, v in sorted(tab.items())})
print(" with workbench letter sets:"); table({k: set(v) for k, v in lt.items()})
print(" with editor's letter sets:"); table({k: set(v) for k, v in lt2.items()})

print("== (c) hashes / listings")
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
for run in sorted(os.listdir(ER)):
    print(run, "top-level:", sorted(os.listdir(ER / run)))
    for f in ("ledger.jsonl", "final_edit.safetensors", "final_edit.json", "run.json"):
        print("   ", f, sha(ER / run / f))
    sd = ER / run / "scripts"
    if sd.exists():
        h = hashlib.sha256()
        for f in sorted(sd.glob("*.py")): h.update(f.name.encode()); h.update(open(f, "rb").read())
        print("    scripts", len(list(sd.glob("*.py"))), "combined", h.hexdigest()[:16])
    for p in sorted((ER / run / "passes").glob("*")) if (ER / run / "passes").exists() else []:
        print("    ", p.name, {f.name: (sha(f) if f.stat().st_size < 2e8 else "big:%d" % f.stat().st_size) for f in sorted(p.iterdir())})
print("sellma root:", sorted(os.listdir(R)))
