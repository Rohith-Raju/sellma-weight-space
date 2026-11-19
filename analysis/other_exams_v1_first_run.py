"""Rotation test outside the Swedish exam, one condition per process (run three in parallel).

Sets, all 4 options with an answer key, prompt layout identical to the bench:
  belebele_sv  Belebele reading comprehension in Swedish (bench's own LÄS prompt, Swedish)
  belebele_en  the SAME Belebele questions in English (the LÄS prompt translated to English)
  mmlu_math    MMLU elementary / high-school / college mathematics (English)
Each question is asked 4 times with the correct answer moved to A, B, C and D (tracked by position).

  uv run --with datasets python analysis/other_exams.py unedited|editor3|random   (first run: from work/)
"""
import json, sys, hashlib, dataclasses
from pathlib import Path
from datasets import load_dataset
from bench.backends import make_backend
from bench.records import Question
from bench.prompts import build_prompt, render_prompt
from bench.editor.evaluate import load_edit, applied, edit_source, random_control
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py

BELEBELE_REV = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
MMLU_REV = "c30699e8356da336a370243923dbaf21066bb9fe"
N_BELEBELE, N_MMLU = 500, 400
condition = sys.argv[1]
out_dir = RESULTS / "experiments/analysis/other_exams"; out_dir.mkdir(parents=True, exist_ok=True)

def first_n(rows, n):
    return sorted(rows, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[:n]

def belebele(lang):
    ds = load_dataset("facebook/belebele", data_files=f"data/{lang}.jsonl", split="train", revision=BELEBELE_REV)
    return {f"{r['link']}#{r['question_number']}": {
        "passage": r["flores_passage"], "q": r["question"],
        "opts": [r[f"mc_answer{i}"] for i in range(1, 5)], "ans": int(r["correct_answer_num"]) - 1} for r in ds}

sv, en = belebele("swe_Latn"), belebele("eng_Latn")
shared = first_n([{"id": k} for k in sv if k in en], N_BELEBELE)
mmlu = []
for subj in ("elementary_mathematics", "high_school_mathematics", "college_mathematics"):
    for i, r in enumerate(load_dataset("cais/mmlu", subj, split="test", revision=MMLU_REV)):
        mmlu.append({"id": f"{subj}/{i}", "q": r["question"], "opts": list(r["choices"]), "ans": int(r["answer"])})
sets = {"belebele_sv": [dict(sv[r["id"]], id=r["id"]) for r in shared],
        "belebele_en": [dict(en[r["id"]], id=r["id"]) for r in shared],
        "mmlu_math": first_n(mmlu, N_MMLU)}

def messages(set_name, item, opts):
    lines = "\n".join(f"{L}. {t}" for L, t in zip("ABCD", opts))
    if set_name == "belebele_sv":   # exactly the bench's LÄS prompt
        q = Question(id=item["id"], exam_date="", exam_pass=0, section="LAS", number=0, question=item["q"],
                     options=dict(zip("ABCD", opts)), answer="A", passage=item["passage"])
        return build_prompt(q)
    if set_name == "belebele_en":   # the same prompt, translated
        text = ("Read the text and answer the question. Choose the option that best agrees with the text.\n\n"
                f"{item['passage']}\n\nQuestion: {item['q']}\n\n{lines}\n\n"
                "Answer with only the letter of the option you choose (A, B, C or D).")
    else:
        text = (f"Choose the correct answer to the question.\n\nQuestion: {item['q']}\n\n{lines}\n\n"
                "Answer with only the letter of the option you choose (A, B, C or D).")
    return [{"role": "user", "content": text}]

backend = make_backend("gemma-4b-full"); tok = backend.tokenizer_path
dummy = Question(id="x", exam_date="", exam_pass=0, section="LAS", number=0, question="", options={}, answer="A")

def run():
    results = {}
    for set_name, items in sets.items():
        per = {}
        for item in items:
            ok = {}
            for shift in range(4):
                opts = item["opts"][shift:] + item["opts"][:shift]
                ans = "ABCD"[(item["ans"] - shift) % 4]          # tracked by position, not by text
                p = backend.letter_probabilities(render_prompt(messages(set_name, item, opts), tok), list("ABCD"), dummy)
                ok[ans] = max("ABCD", key=lambda L: p.get(L, 0.0)) == ans
            per[item["id"]] = ok
        results[set_name] = per
        n = len(per)
        print(condition, set_name, {L: round(100 * sum(v[L] for v in per.values()) / n, 1) for L in "ABCD"},
              "all4", sum(all(v.values()) for v in per.values()), flush=True)
    return results

edit = load_edit(edit_source("trial_b50_k"))
if condition == "unedited":
    results = run()
elif condition == "editor3":
    with applied(backend, edit): results = run()
else:
    with applied(backend, random_control(edit)): results = run()
backend.close()
json.dump({"condition": condition, "revisions": {"belebele": BELEBELE_REV, "mmlu": MMLU_REV}, "results": results},
          open(out_dir / f"{condition}.json", "w"))
print(condition, "DONE", flush=True)
