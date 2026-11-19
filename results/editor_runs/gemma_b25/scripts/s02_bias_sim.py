from bench.editor import Workbench
import numpy as np
wb = Workbench.open("gemma_b25")
P = wb.practice_questions()
L="ABCDE"
def acc(items, bias):
    c=0
    for p in items:
        lp={l:np.log(max(v,1e-30))+bias.get(l,0) for l,v in p.baseline.letter_probs.items()}
        c+= max(lp,key=lp.get)==p.question.answer
    return c
for sec in ["ORD","LAS","MEK"]:
    for paper in ["drill","validation"]:
        it=[p for p in P if p.question.section==sec and p.paper==paper]
        print(sec,paper,len(it),[ (b,acc(it,{"A":b})) for b in [0,0.5,1,1.5,2,2.5,3,4,5]])
    it=[p for p in P if p.question.section==sec]
    # mean log prob per letter, and mean prob
    for l in L:
        v=[p.baseline.letter_probs.get(l) for p in it if l in p.baseline.letter_probs]
        if v: print("  ",l,"mean p %.3f"%np.mean(v),"mean logp %.2f"%np.mean(np.log(np.maximum(v,1e-30))))
    # when key=A and wrong, margin needed
    m=[np.log(max(p.baseline.letter_probs.values()))-np.log(max(p.baseline.letter_probs["A"],1e-30)) for p in it if p.question.answer=="A" and not p.baseline.is_correct]
    print("  key A wrong margins", np.round(sorted(m),1))
