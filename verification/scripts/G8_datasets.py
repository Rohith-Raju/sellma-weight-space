"""G8 verifier: dataset identity, selection, rotation tracking and prompt layout.

Uses Belebele / MMLU files downloaded at the pinned revisions with curl (directory passed as argv[1]):
  B=https://huggingface.co/datasets/facebook/belebele/resolve/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2
  M=https://huggingface.co/datasets/cais/mmlu/resolve/c30699e8356da336a370243923dbaf21066bb9fe
  $B/data/{swe_Latn,eng_Latn}.jsonl, $M/{elementary,high_school,college}_mathematics/test-00000-of-00001.parquet
Run: uv run --with pyarrow python -I verification/scripts/G8_datasets.py <download_dir>
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, _REPO + "/src")
from bench.prompts import build_prompt  # noqa: E402  (no transformers needed for build_prompt)
from bench.records import Question  # noqa: E402

DL = Path(sys.argv[1])
OE = Path(_RES + "/experiments/analysis/other_exams")
saved = json.loads((OE / "unedited.json").read_text())["results"]


def sha_order(ids):
    return sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())


def bb(lang):
    rows = [json.loads(l) for l in (DL / f"{lang}.jsonl").read_text().splitlines() if l.strip()]
    out = {}
    for r in rows:
        k = f"{r['link']}#{r['question_number']}"
        assert k not in out, f"duplicate id {k}"
        out[k] = r
    return rows, out


sv_rows, sv = bb("swe_Latn")
en_rows, en = bb("eng_Latn")
print(f"Belebele rows: sv {len(sv_rows)}, en {len(en_rows)}; unique ids sv {len(sv)}, en {len(en)}; shared {len(set(sv) & set(en))}")
shared = [k for k in sv if k in en]
mine = sha_order(shared)[:500]
print("first-500-by-sha256 of shared ids == saved sv id list (as set):", set(mine) == set(saved["belebele_sv"]),
      "| same order:", mine == list(saved["belebele_sv"]))
print("saved en ids == saved sv ids:", list(saved["belebele_en"]) == list(saved["belebele_sv"]))
same_key = sum(sv[k]["correct_answer_num"] == en[k]["correct_answer_num"] for k in mine)
same_passage_id = sum(sv[k].get("passage_id") == en[k].get("passage_id") for k in mine) if "passage_id" in sv_rows[0] else "n/a"
print(f"of the 500: same correct_answer_num sv vs en {same_key}/500; same passage_id {same_passage_id}")
print("answer key distribution (sv, 500):", Counter(sv[k]["correct_answer_num"] for k in mine))
print("example sv/en pair:", mine[0], "|", sv[mine[0]]["question"][:70], "|", en[mine[0]]["question"][:70])
ws = sum(any(s != s.strip() for s in [r["flores_passage"], r["question"]] + [r[f"mc_answer{i}"] for i in range(1, 5)])
         for r in (en[k] for k in mine))
print(f"English items with leading/trailing whitespace in passage/question/options (stripped in sv, not in en): {ws}")
dup = sum(len({sv[k][f'mc_answer{i}'] for i in range(1, 5)}) < 4 for k in mine)
print(f"sv items with duplicate option text: {dup}")

# MMLU
mm = []
for subj in ("elementary_mathematics", "high_school_mathematics", "college_mathematics"):
    t = pq.read_table(DL / f"{subj}.parquet").to_pylist()
    print(f"MMLU {subj}: {len(t)} test rows; choice counts {Counter(len(r['choices']) for r in t)}")
    mm += [{"id": f"{subj}/{i}", **r} for i, r in enumerate(t)]
mine_m = sha_order([r["id"] for r in mm])[:400]
print("MMLU first-400-by-sha256 == saved id list:", mine_m == list(saved["mmlu_math"]))
print("MMLU subject mix of the 400:", Counter(i.split("/")[0] for i in mine_m))
mmd = {r["id"]: r for r in mm}
print("MMLU answer key distribution (400):", Counter("ABCD"[mmd[i]["answer"]] for i in mine_m))

# Rotation tracking (replica of other_exams.py logic): the correct TEXT must sit at the claimed letter.
errs = 0
for k in mine:
    opts0 = [sv[k][f"mc_answer{i}"] for i in range(1, 5)]
    a0 = int(sv[k]["correct_answer_num"]) - 1
    seen = set()
    for shift in range(4):
        opts = opts0[shift:] + opts0[:shift]
        ans = "ABCD"[(a0 - shift) % 4]
        seen.add(ans)
        errs += opts["ABCD".index(ans)] != opts0[a0]
    errs += seen != set("ABCD")
print("rotation tracking errors over 500 sv items:", errs)

# Prompt layout: bench LAS prompt vs the English template in other_exams.py
k = mine[0]
r = sv[k]
opts = [r[f"mc_answer{i}"] for i in range(1, 5)]
q = Question(id=k, exam_date="", exam_pass=0, section="LAS", number=0, question=r["question"],
             options=dict(zip("ABCD", opts)), answer="A", passage=r["flores_passage"])
sv_prompt = build_prompt(q)[0]["content"]
e = en[k]
eopts = [e[f"mc_answer{i}"] for i in range(1, 5)]
lines = "\n".join(f"{L}. {t}" for L, t in zip("ABCD", eopts))
en_prompt = ("Read the text and answer the question. Choose the option that best agrees with the text.\n\n"
             f"{e['flores_passage']}\n\nQuestion: {e['question']}\n\n{lines}\n\n"
             "Answer with only the letter of the option you choose (A, B, C or D).")
print("\n--- sv prompt (bench build_prompt) ---\n" + sv_prompt)
print("\n--- en prompt (other_exams.py template) ---\n" + en_prompt)
print("\nblock count sv/en:", len(sv_prompt.split("\n\n")), len(en_prompt.split("\n\n")))
