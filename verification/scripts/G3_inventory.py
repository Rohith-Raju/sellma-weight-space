"""G3 verification: inventory every run folder of the single-question edit archive and recompute
headline (LAS+MEK) accuracy, letter shares, perplexity and target outcomes from the raw files.

Read-only. Run from the project root:  uv run python -I verification/scripts/G3_inventory.py
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(_REPO + "")
ARCH = Path(_RES) / "archive/single_question_edits"
QUESTIONS = ROOT / "exams/questions.jsonl"
SPLIT = ROOT / "exams/split.json"
HEAD = ("LAS", "MEK")


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


questions = {q["id"] if "id" in q else q["question_id"]: q for q in load_jsonl(QUESTIONS)}
split = json.load(open(SPLIT))


def keyed(qid):
    q = questions[qid]
    for k in ("correct_letter", "answer", "correct", "key"):
        if k in q:
            return q[k]
    raise KeyError(qid)


def run_info(d):
    rj = d / "run.json"
    info = {"path": str(d.relative_to(ARCH)), "has_run_json": rj.exists()}
    run = json.load(open(rj)) if rj.exists() else {}
    info["finished"] = bool(run.get("finished_at"))
    info["n_expected"] = run.get("n_questions")
    ans_p = d / "answers.jsonl"
    ans = load_jsonl(ans_p) if ans_p.exists() else []
    info["n_answers"] = len(ans)
    ids = [a["question_id"] for a in ans]
    info["dup_ids"] = len(ids) - len(set(ids))
    # key check against questions.jsonl (independent of the answers file's own correct_letter)
    mism = 0
    for a in ans:
        if a["question_id"] in questions and keyed(a["question_id"]) != a["correct_letter"]:
            mism += 1
        # is_correct consistency
        if (a["chosen_letter"] == a["correct_letter"]) != a["is_correct"]:
            mism += 1000
    info["key_mismatch"] = mism
    h = [a for a in ans if a["section"] in HEAD]
    info["n_head"] = len(h)
    info["head_correct"] = sum(a["chosen_letter"] == keyed(a["question_id"]) for a in h)
    info["head_acc"] = 100 * info["head_correct"] / len(h) if h else None
    info["all_correct"] = sum(a["chosen_letter"] == keyed(a["question_id"]) for a in ans)
    c = Counter(a["chosen_letter"] for a in h)
    info["head_letter_share"] = {k: round(100 * c[k] / len(h), 1) for k in "ABCD"} if h else {}
    info["head_ids"] = sorted(a["question_id"] for a in h)
    info["answers"] = {a["question_id"]: a for a in ans}
    dmg = d / "damage.json"
    if dmg.exists():
        dj = json.load(open(dmg))
        info["ppl"] = dj.get("perplexity")
        info["loop"] = dj.get("loop_rate_percent")
    if run.get("experiment"):
        info["edits"] = run["experiment"].get("edits")
    info["model"] = run.get("model")
    info["split"] = run.get("split")
    info["started_at"] = run.get("started_at")
    info["finished_at"] = run.get("finished_at")
    return info


def main():
    rows = []
    for rj in sorted(ARCH.rglob("run.json")):
        rows.append(run_info(rj.parent))
    # also folders with answers but no run.json
    for a in sorted(ARCH.rglob("answers.jsonl")):
        if not (a.parent / "run.json").exists():
            rows.append(run_info(a.parent))
    test_ids = set()
    for k, v in split.items():
        pass
    print(f"{'folder':70s} fin  n/exp   head  acc%   ppl     loop  shares(A B C D)")
    for r in rows:
        print(f"{r['path']:70s} {'Y' if r['finished'] else 'N'}  {r['n_answers']:>3}/{r['n_expected']!s:>3} "
              f"{r['head_correct']:>3}/{r['n_head']:<3} {r['head_acc'] if r['head_acc'] is None else round(r['head_acc'],1)!s:>5} "
              f"{r.get('ppl') and round(r['ppl'],3)!s:>7} {r.get('loop')!s:>5} "
              f"{' '.join(str(r['head_letter_share'].get(k)) for k in 'ABCD')}"
              f"{'  DUP' if r['dup_ids'] else ''}{'  KEYMISMATCH=' + str(r['key_mismatch']) if r['key_mismatch'] else ''}")
    json.dump([{k: v for k, v in r.items() if k != "answers"} for r in rows],
              open(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/dev/null"), "w"), indent=1)


if __name__ == "__main__":
    main()
