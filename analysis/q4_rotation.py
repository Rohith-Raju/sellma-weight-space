"""Rotation test at 4-bit (Q4_K_M), unedited vs editor 3's change, on the held-out hidden + test LÄS/MEK
questions. Same rotation logic as heldout_position.py (answer tracked by position). Runs llama.cpp b8680
(the server build) on the GGUFs made by q4_weight_check.py. Questions are split into shards so
several servers can run in parallel; each shard appends to its own file.

  Runs on the GPU server:  uv run python -I analysis/q4_rotation.py unedited|edited|control SHARD NSHARDS
"""
import json, sys, dataclasses
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bench.backends.llamacpp import LlamaCppBackend        # noqa: E402
from bench.records import load_questions                   # noqa: E402
from bench.prompts import build_prompt, render_prompt      # noqa: E402

which = sys.argv[1]
shard, nshards = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
DATA = ROOT / "gpu_results" if (ROOT / "gpu_results").exists() else ROOT
# Input: the group split from DATA if it is there, else the saved copy in results/ (fresh clone).
# Output stays under DATA (the repository root on a clone), so results/ is never overwritten.
SPLIT_SRC = DATA if (DATA / "experiments/group/gemma-4b-full/split.json").exists() else ROOT / "results"
gguf = ROOT / f"work/q4check/{which}-Q4_K_M.gguf"
tok = str(ROOT / "models/gemma-3-4b-it-official")
split = json.load(open(SPLIT_SRC / "experiments/group/gemma-4b-full/split.json"))["question_ids"]
hidden = set(split["hidden_wrong"]) | set(split["hidden_right"])
test = set(json.load(open(ROOT / "exams/split.json"))["splits"]["test"]["question_ids"])
qs = {q.id: q for q in load_questions(ROOT / "exams/questions.jsonl")}
sets = {"hidden": [qs[i] for i in sorted(hidden) if qs[i].section in ("LAS", "MEK")],
        "test": [qs[i] for i in sorted(test) if qs[i].section in ("LAS", "MEK")]}
sets = {k: [q for i, q in enumerate(v) if i % nshards == shard] for k, v in sets.items()}
out = DATA / f"experiments/analysis/q4_rotation/{which}_shard{shard}of{nshards}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
backend = LlamaCppBackend(gguf_path=gguf, tokenizer_path=tok)   # server build, default bin dir
with open(out, "a") as f:
    for set_name, questions in sets.items():
        for q in questions:
            if q.id in done:
                continue
            letters = "ABCD"; ci = letters.index(q.answer); texts = [q.options[L] for L in letters]
            ok, chosen = {}, {}
            for shift in range(4):
                rot = texts[shift:] + texts[:shift]; ans = letters[(ci - shift) % 4]
                rq = dataclasses.replace(q, options=dict(zip(letters, rot)), answer=ans)
                p = backend.letter_probabilities(render_prompt(build_prompt(rq), tok), list(letters), rq)
                c = max(letters, key=lambda L: p.get(L, 0.0)); ok[ans] = c == ans; chosen[ans] = c
            f.write(json.dumps({"set": set_name, "id": q.id, "section": q.section, "answer": q.answer,
                                "correct_by_position": ok, "chosen_by_position": chosen,
                                "original_order_correct": ok[q.answer]}) + "\n"); f.flush()
backend.close()
print(which, shard, "DONE")
