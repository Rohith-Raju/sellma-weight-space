import json, sys
from _paths import ROOT, RESULTS  # project folders; see analysis/_paths.py
sys.path.insert(0, str(ROOT / "src"))
from bench.editor import Workbench
for p in sys.argv[1:]:
    run, folder = p.split("/passes/")
    wb = Workbench.open(run)
    practice = {q.id if hasattr(q, "id") else q["id"]: q for q in wb.practice_questions()}
    req = json.load(open(p + "/request.json")); res = json.load(open(p + "/result.json"))
    ids = req.get("question_ids") or req.get("ids") or [x.get("id") for x in req.get("questions", [])]
    probs = res["letter_probs"]
    fixed = broke = edited_ok = base_ok = 0
    for qid, lp in zip(ids, probs):
        q = practice[qid]
        ans = getattr(q, "answer", None) or q.question.answer
        base_right = getattr(q, "baseline_correct", None)
        if base_right is None:
            b = getattr(q, "baseline", None) or {}
            base_right = b.get("is_correct") if isinstance(b, dict) else getattr(b, "is_correct", None)
        chosen = max(lp, key=lp.get) if isinstance(lp, dict) else "ABCDE"[max(range(len(lp)), key=lambda i: lp[i])]
        ok = chosen == ans
        edited_ok += ok; base_ok += bool(base_right)
        fixed += ok and not base_right; broke += (not ok) and bool(base_right)
    print(p, f"n={len(ids)} correct before {base_ok} -> after {edited_ok} (fixed {fixed}, broke {broke})")
