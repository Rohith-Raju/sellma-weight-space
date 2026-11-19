from bench.editor import Workbench
import numpy as np
wb = Workbench.open("gemma_b10")
P = wb.practice_questions()
def acc(items, bias):
    n=0
    for p in items:
        lp={k:np.log(max(v,1e-30))+bias.get(k,0) for k,v in p.baseline.letter_probs.items()}
        n+= max(lp,key=lp.get)==p.question.answer
    return n/len(items)
for sec in ["ORD","LAS","MEK"]:
    for paper in ["drill","validation"]:
        it=[p for p in P if p.question.section==sec and p.paper==paper]
        print(sec,paper,[ (b,round(acc(it,{"A":b}),3)) for b in [0,0.5,1,1.5,2,3,4,5]])
# logit margins: distribution of log p(top)/p(second)
for sec in ["ORD","LAS","MEK"]:
    it=[p for p in P if p.question.section==sec]
    m=[]
    for p in it:
        v=sorted(p.baseline.letter_probs.values(),reverse=True); m.append(np.log(v[0]/max(v[1],1e-30)))
    print(sec,"margin quantiles",np.round(np.quantile(m,[.1,.25,.5,.75,.9]),2))
    # mean log-prob of each letter when it is NOT key (bias estimate)
    for L in "ABCDE":
        x=[np.log(max(p.baseline.letter_probs[L],1e-30)) for p in it if L in p.baseline.letter_probs and p.question.answer!=L]
        if x: print("   ",L,"mean logp when not key %.2f"%np.mean(x), "median %.2f"%np.median(x))
# per-letter best bias fit on drill, eval on validation
import itertools
for sec in ["LAS","MEK"]:
    d=[p for p in P if p.question.section==sec and p.paper=="drill"]; v=[p for p in P if p.question.section==sec and p.paper=="validation"]
    best=max(itertools.product([0,1,2,3],[ -1,0,1],[-1,0,1],[-1,0,1]),key=lambda b:acc(d,dict(zip("ABCD",b))))
    print(sec,"best",best,acc(d,dict(zip("ABCD",best))),acc(v,dict(zip("ABCD",best))))
