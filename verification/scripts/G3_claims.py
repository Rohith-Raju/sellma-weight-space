"""G3 verification: check each G3 claim against the raw archive files.

Read-only. Run from the project root:  uv run python -I verification/scripts/G3_claims.py
Only run folders whose run.json has finished_at are treated as complete.
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import json
from collections import Counter
from pathlib import Path

ROOT = Path(_REPO + "")
ARCH = Path(_RES) / "archive/single_question_edits"
HEAD = ("LAS", "MEK")

questions = {}
for line in open(ROOT / "exams/questions.jsonl"):
    if line.strip():
        q = json.loads(line)
        questions[q["id"]] = q
split = json.load(open(ROOT / "exams/split.json"))
test_dates = set(split["config"]["test"])
test_head_ids = {i for i, q in questions.items() if q["exam_date"] in test_dates and q["section"] in HEAD}


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def run(d):
    rj = d / "run.json"
    if not rj.exists():
        return None
    r = json.load(open(rj))
    ans = load_jsonl(d / "answers.jsonl") if (d / "answers.jsonl").exists() else []
    dmg = json.load(open(d / "damage.json")) if (d / "damage.json").exists() else {}
    return {"finished": bool(r.get("finished_at")), "n_q": r.get("n_questions"), "ans": ans, "dmg": dmg,
            "run": r}


def head(ans):
    h = [a for a in ans if a["section"] in HEAD]
    corr = sum(a["chosen_letter"] == questions[a["question_id"]]["answer"] for a in h)
    share = Counter(a["chosen_letter"] for a in h)
    return corr, len(h), {k: 100 * share[k] / len(h) for k in "ABCD"} if h else {}, {a["question_id"] for a in h}


def fmt(c, n):
    return f"{100 * c / n:.1f} ({c}/{n})" if n else "n/a"


for model in ("gemma-4b-full", "qwen-9b-full"):
    print("=" * 100)
    print(model)
    fc = load_jsonl(ARCH / model / "failure_cases.jsonl")
    print(f"failure cases: {len(fc)}  sections: {Counter(f.get('section') for f in fc)}")
    base = run(ARCH / model / "baseline/full_test")
    bc, bn, bshare, bids = head(base["ans"])
    assert bids == test_head_ids, "baseline test headline ids != LAS+MEK of the 3 test exams"
    print(f"baseline test headline {fmt(bc, bn)} finished={base['finished']} ppl={base['dmg'].get('perplexity')!r} "
          f"loop={base['dmg'].get('loop_rate_percent')} share={ {k: round(v, 1) for k, v in bshare.items()} }")
    deltas = []
    for f in fc:
        qid = f["question_id"]
        folder = ARCH / model / "edits" / qid.replace("/", "_")
        line = f"  {qid:24s}"
        if not folder.exists():
            print(line + " NO FOLDER (edit never reached)")
            continue
        s = json.load(open(folder / "search.json"))
        tried = [(t["alpha_fraction"], t["is_correct"]) for t in s.get("tried", [])]
        af = s["edit"]["alpha_fraction"] if s.get("edit") else None
        line += f" status={s.get('status'):10s} fixed={s.get('fixed')!s:5s} alpha_frac={af} tried={tried}"
        print(line)
        for kind in ("targeted_target", "targeted_test", "control_target", "control_test", "targeted_dev_subset",
                     "control_dev_subset", "targeted_q4_target", "targeted_q4_test"):
            r = run(folder / kind)
            if r is None:
                if (folder / kind).exists():
                    print(f"      {kind:20s} folder without run.json")
                continue
            c, n, share, ids = head(r["ans"])
            tag = "complete" if r["finished"] else "INCOMPLETE (no finished_at)"
            extra = ""
            if kind.endswith("_test"):
                if ids != test_head_ids and r["finished"]:
                    extra += " IDS!=test"
                extra += (f" ppl={r['dmg'].get('perplexity')!r} loop={r['dmg'].get('loop_rate_percent')}"
                          f" share={ {k: round(v, 1) for k, v in share.items()} }")
            if kind.endswith("target"):
                a = [x for x in r["ans"] if x["question_id"] == qid]
                extra += f" target chosen={a[0]['chosen_letter'] if a else None} key={questions[qid]['answer']}"
            print(f"      {kind:20s} {tag:28s} n={len(r['ans'])}/{r['n_q']} head={fmt(c, n)}{extra}")
            if kind == "targeted_test" and r["finished"]:
                deltas.append((qid, c - bc))
    if deltas:
        mean_q = sum(d for _, d in deltas) / len(deltas)
        print(f"  targeted_test deltas vs baseline (questions): {deltas}")
        print(f"  mean change = {mean_q:.3f} questions = {100 * mean_q / bn:.3f} points over {len(deltas)} edits")
