from bench.editor import Workbench
from collections import Counter, defaultdict
import numpy as np
wb = Workbench.open("gemma_b10")
P = wb.practice_questions()
print(len(P), wb.budget, wb.passes_left)
for sec in ["ORD","LAS","MEK"]:
    for paper in ["drill","validation"]:
        it=[p for p in P if p.question.section==sec and p.paper==paper]
        if not it: continue
        acc=np.mean([p.baseline.is_correct for p in it])
        nopt=Counter(len(p.question.options) for p in it)
        print(sec,paper,len(it),"acc %.3f"%acc,"nopt",dict(nopt),
              "probletters %.3f"%np.mean([p.baseline.prob_on_letters for p in it]))
for sec in ["ORD","LAS","MEK"]:
    it=[p for p in P if p.question.section==sec]
    key=Counter(p.question.answer for p in it); ch=Counter(p.baseline.chosen_letter for p in it)
    wrongch=Counter(p.baseline.chosen_letter for p in it if not p.baseline.is_correct)
    wrongkey=Counter(p.question.answer for p in it if not p.baseline.is_correct)
    print(sec,"key",sorted(key.items()),"\n   chosen",sorted(ch.items()),"\n   wrong-chosen",sorted(wrongch.items()),"\n   wrong-key",sorted(wrongkey.items()))
    # per-key accuracy
    print("   acc by key", {k: round(np.mean([p.baseline.is_correct for p in it if p.question.answer==k]),3) for k in sorted(key)})
    # confidence
    pc=[max(p.baseline.letter_probs.values())/max(p.baseline.prob_on_letters,1e-9) for p in it]
    w=[max(p.baseline.letter_probs.values())/max(p.baseline.prob_on_letters,1e-9) for p in it if not p.baseline.is_correct]
    c=[max(p.baseline.letter_probs.values())/max(p.baseline.prob_on_letters,1e-9) for p in it if p.baseline.is_correct]
    print("   maxprob mean all %.3f correct %.3f wrong %.3f"%(np.mean(pc),np.mean(c),np.mean(w)))
    # rank of correct among wrong
    rk=Counter()
    for p in it:
        if p.baseline.is_correct: continue
        lp=p.baseline.letter_probs; s=sorted(lp,key=lambda k:-lp[k]); rk[s.index(p.question.answer)+1]+=1
    print("   rank of key when wrong",sorted(rk.items()))
    # prompt length for LAS
    if sec=="LAS":
        L=[len(p.question.passage or "") for p in it]
        import statistics
        med=statistics.median(L)
        print("   passage len median",med,"acc short %.3f long %.3f"%(np.mean([p.baseline.is_correct for p in it if len(p.question.passage or "")<=med]),np.mean([p.baseline.is_correct for p in it if len(p.question.passage or "")>med])))
print("=====PROMPTS")
for sec in ["ORD","MEK"]:
    it=[p for p in P if p.question.section==sec][0]
    print(repr(wb.render_prompt(it)))
it=[p for p in P if p.question.section=="LAS"][0]
t=wb.render_prompt(it); print(repr(t[:400])); print("...."); print(repr(t[-900:]))
