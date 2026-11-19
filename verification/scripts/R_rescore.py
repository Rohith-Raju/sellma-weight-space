"""Independent replication R: re-score random samples with plain transformers and compare with saved files.

Does not import bench.backends / bench.run / bench.editor. Uses bench.prompts (prompt text) and
bench.records.Question (plain dataclass) only. Run: cd /root/sellma && uv run python verification/R_rescore.py
"""
import dataclasses, glob, hashlib, json, random, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from bench.prompts import build_prompt, render_prompt
from bench.records import Question

ROOT = Path("/root/sellma")
MODEL = ROOT / "models/gemma-3-4b-it-official"
SEED = 20261119
OUT = ROOT / "verification/R_results.json"
report = {"seed": SEED}

# ---- 0. weight checksums
expected = {"model-00001-of-00002.safetensors": "eb5fd5e97d", "model-00002-of-00002.safetensors": "fdde0e5aa5"}
sha = {}
for name, prefix in expected.items():
    h = hashlib.sha256()
    with open(MODEL / name, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    sha[name] = h.hexdigest()
    print("sha256", name, sha[name], "OK" if sha[name].startswith(prefix) else "MISMATCH", flush=True)
report["sha256"] = sha
report["sha256_ok"] = all(sha[n].startswith(p) for n, p in expected.items())

# ---- model + my own scorer
tok = AutoTokenizer.from_pretrained(str(MODEL))
model = Gemma3ForConditionalGeneration.from_pretrained(str(MODEL), dtype=torch.bfloat16, device_map="cuda").eval()
letter_ids = {L: [] for L in "ABCDE"}
for i in range(len(tok)):
    t = tok.decode([i]).strip()
    if t in letter_ids:
        letter_ids[t].append(i)
print({L: len(v) for L, v in letter_ids.items()}, flush=True)

def score(messages, letters):
    prompt = render_prompt(messages, str(MODEL))
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    with torch.inference_mode():
        logits = model(input_ids=ids).logits[0, -1]      # full logits, last position (no logits_to_keep)
        p = torch.softmax(logits.float(), dim=-1).double().cpu()
    return {L: float(p[[i for i in letter_ids[L] if i < len(p)]].sum()) for L in letters}

def pick(p, letters):
    return max(letters, key=lambda L: p.get(L, 0.0))

questions = {}
for line in open(ROOT / "exams/questions.jsonl", encoding="utf-8"):
    if line.strip():
        q = Question(**json.loads(line)); questions[q.id] = q

# ---- 1. standard test run, original order
run_dirs = sorted(glob.glob(str(ROOT / "runs/*_gemma-4b-full_test")))
assert len(run_dirs) == 1, run_dirs
saved = [json.loads(l) for l in open(Path(run_dirs[0]) / "answers.jsonl")]
sample = random.Random(SEED).sample(saved, 40)
p1 = {"run": run_dirs[0], "n": len(sample), "letter_agree": 0, "prob_agree_1e-3": 0, "max_abs_prob_diff": 0.0, "mismatches": [], "rows": []}
for s in sample:
    q = questions[s["question_id"]]
    letters = sorted(q.options)
    p = score(build_prompt(q), letters)
    chosen = pick(p, letters)
    diff = max(abs(p[L] - s["letter_probs"][L]) for L in letters)
    p1["max_abs_prob_diff"] = max(p1["max_abs_prob_diff"], diff)
    p1["letter_agree"] += chosen == s["chosen_letter"]
    p1["prob_agree_1e-3"] += diff <= 1e-3
    row = {"id": q.id, "saved": s["chosen_letter"], "mine": chosen, "max_abs_prob_diff": diff}
    p1["rows"].append(row)
    if chosen != s["chosen_letter"] or diff > 1e-3:
        p1["mismatches"].append({**row, "saved_probs": s["letter_probs"], "my_probs": p})
print("PART1", {k: v for k, v in p1.items() if k != "rows"}, flush=True)
report["part1_test_run"] = p1

# ---- 2/3. held-out rotations
held = [json.loads(l) for l in open(ROOT / "experiments/analysis/heldout_position.jsonl")]
hidden = {c: {r["id"]: r for r in held if r["condition"] == c and r["set"] == "hidden"} for c in ("unedited", "editor3_edit")}
ids30 = random.Random(SEED + 1).sample(sorted(hidden["unedited"]), 30)

def rotation_part(condition):
    out = {"n_questions": 30, "n_cells": 0, "cells_agree": 0, "original_order_n": 0, "original_order_agree": 0,
           "option_average_agree": 0, "mismatches": [], "rows": []}
    for qid in ids30:
        q, s = questions[qid], hidden[condition][qid]
        texts = [q.options[L] for L in "ABCD"]; correct = q.options[q.answer]
        mine, detail, per_text = {}, {}, {t: 0.0 for t in texts}
        for shift in range(4):
            rot = texts[shift:] + texts[:shift]
            ans = "ABCD"[rot.index(correct)]
            p = score(build_prompt(dataclasses.replace(q, options=dict(zip("ABCD", rot)), answer=ans)), list("ABCD"))
            tot = sum(p.values())
            for L, t in zip("ABCD", rot): per_text[t] += p[L] / tot
            mine[ans] = pick(p, "ABCD") == ans
            detail[ans] = {"chosen": pick(p, "ABCD"), "probs": p}
        avg_ok = max(texts, key=lambda t: per_text[t]) == correct
        orig_ok = mine[q.answer]   # shift 0 is the original order
        for L in "ABCD":
            out["n_cells"] += 1
            ok = mine[L] == s["correct_by_position"][L]
            out["cells_agree"] += ok
            if not ok:
                out["mismatches"].append({"id": qid, "answer_at": L, "saved_correct": s["correct_by_position"][L],
                                          "my_correct": mine[L], **detail[L]})
        out["original_order_n"] += 1
        out["original_order_agree"] += orig_ok == s["original_order_correct"]
        out["option_average_agree"] += avg_ok == s["option_average_correct"]
        if orig_ok != s["original_order_correct"]:
            out["mismatches"].append({"id": qid, "field": "original_order_correct", "saved": s["original_order_correct"], "mine": orig_ok})
        if avg_ok != s["option_average_correct"]:
            out["mismatches"].append({"id": qid, "field": "option_average_correct", "saved": s["option_average_correct"], "mine": avg_ok})
        out["rows"].append({"id": qid, "saved": s["correct_by_position"], "mine": mine})
    return out

p2 = rotation_part("unedited")
print("PART2", {k: v for k, v in p2.items() if k != "rows"}, flush=True)
report["part2_heldout_unedited"] = p2

# editor 3's change: read the file directly
edit_path = ROOT / "editor_runs/trial_b50_k/final_edit.safetensors"
with safe_open(str(edit_path), framework="pt") as f:
    meta = f.metadata(); tensors = {k: f.get_tensor(k) for k in f.keys()}
forms = json.loads(meta["forms"])
print("edit keys", {k: (tuple(v.shape), str(v.dtype)) for k, v in tensors.items()}, "forms", forms, flush=True)
params = dict(model.named_parameters())
backups, edit_info = {}, {"forms": forms, "tensors": {}}
with torch.no_grad():
    for name, form in forms.items():
        w = params[name].data
        backups[name] = w.clone()
        values = tensors[f"values:{name}"].to(device=w.device, dtype=torch.float32)
        if form == "dense":
            w.copy_((w.float() + values).to(w.dtype))
        elif form == "rows":
            idx = tensors[f"rows:{name}"].to(w.device).long()
            w[idx] = (w[idx].float() + values).to(w.dtype)
        elif form == "columns":
            idx = tensors[f"columns:{name}"].to(w.device).long()
            w[:, idx] = (w[:, idx].float() + values).to(w.dtype)
        else:
            raise ValueError(form)
        changed = int((w != backups[name]).sum())
        edit_info["tensors"][name] = {"form": form, "values_shape": list(values.shape), "weights_changed_in_bf16": changed,
                                      "weights_touched": int(values.numel())}
print("edit applied", edit_info, flush=True)
report["edit"] = edit_info
try:
    p3 = rotation_part("editor3_edit")
finally:
    with torch.no_grad():
        for name, b in backups.items():
            params[name].data.copy_(b)
            assert torch.equal(params[name].data, b)
print("PART3", {k: v for k, v in p3.items() if k != "rows"}, flush=True)
report["part3_heldout_editor3"] = p3

# sanity: weights restored -> first part-1 question gives the same probabilities again
q0 = questions[sample[0]["question_id"]]
again = score(build_prompt(q0), sorted(q0.options))
report["restored_check_max_diff"] = max(abs(again[L] - sample[0]["letter_probs"][L]) for L in again)
print("restored check diff", report["restored_check_max_diff"], flush=True)

# ---- 4. Belebele English
BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
saved_oe = json.load(open(ROOT / "experiments/analysis/other_exams/unedited_with_choices.json"))["results"]
path = hf_hub_download("facebook/belebele", "data/eng_Latn.jsonl", repo_type="dataset", revision=BELEBELE_REV)
items = {}
for line in open(path, encoding="utf-8"):
    r = json.loads(line)
    items[f"{r['link']}#{r['question_number']}"] = r
ids20 = random.Random(SEED + 2).sample(sorted(saved_oe["belebele_en"]), 20)
p4 = {"n_items": 20, "n_cells": 0, "chosen_agree": 0, "correct_agree": 0, "mismatches": [], "rows": []}
for iid in ids20:
    r = items[iid]
    base = [r[f"mc_answer{i}"] for i in range(1, 5)]; ans_idx = int(r["correct_answer_num"]) - 1
    mine = {}
    for shift in range(4):
        opts = base[shift:] + base[:shift]
        ans = "ABCD"[(ans_idx - shift) % 4]
        lines = "\n".join(f"{L}. {t}" for L, t in zip("ABCD", opts))
        text = ("Read the text and answer the question. Choose the option that best agrees with the text.\n\n"
                f"{r['flores_passage']}\n\nQuestion: {r['question']}\n\n{lines}\n\n"
                "Answer with only the letter of the option you choose (A, B, C or D).")
        p = score([{"role": "user", "content": text}], list("ABCD"))
        chosen = pick(p, "ABCD"); mine[ans] = chosen
        s_chosen = saved_oe["belebele_en/chosen"][iid][ans]; s_ok = saved_oe["belebele_en"][iid][ans]
        p4["n_cells"] += 1
        p4["chosen_agree"] += chosen == s_chosen
        p4["correct_agree"] += (chosen == ans) == s_ok
        if chosen != s_chosen or (chosen == ans) != s_ok:
            p4["mismatches"].append({"id": iid, "answer_at": ans, "saved_chosen": s_chosen, "saved_correct": s_ok,
                                     "my_chosen": chosen, "my_probs": p})
    p4["rows"].append({"id": iid, "saved": saved_oe["belebele_en/chosen"][iid], "mine": mine})
print("PART4", {k: v for k, v in p4.items() if k != "rows"}, flush=True)
report["part4_belebele_en"] = p4

json.dump(report, open(OUT, "w"), indent=2, ensure_ascii=False)
print("WROTE", OUT)
