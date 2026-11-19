from bench.editor import Workbench
from collections import Counter
import numpy as np, itertools
wb = Workbench.open("gemma_b50")
P = wb.practice_questions()
L="ABCDE"
def logp(p):
    return np.array([np.log(max(p.baseline.letter_probs.get(l,0),1e-30)) for l in L[:len(p.question.options)]])
for sec in ["ORD","LAS","MEK"]:
    it=[p for p in P if p.question.section==sec]
    X=np.stack([logp(p) for p in it]); y=np.array([L.index(p.question.answer) for p in it])
    n=X.shape[1]
    print(sec,"base",(X.argmax(1)==y).mean(), "mean logp per letter",X.mean(0).round(2), "median",np.median(X,0).round(2))
    # mean logp of letter when it is NOT key and when it IS key
    for k in range(n):
        print("  ",L[k],"as key: acc",(X[y==k].argmax(1)==k).mean().round(3),"n",(y==k).sum(),"| median logp when not key",np.median(X[y!=k][:,k]).round(2), "| median margin when key (lp_key - max other)",np.median(X[y==k][:,k]-np.delete(X[y==k],k,1).max(1)).round(2))
    for b in [0.5,1,2,3,4,6,8,10,14]:
        Xb=X.copy(); Xb[:,0]+=b
        dr=np.array([p.paper=="drill" for p in it])
        acc=(Xb.argmax(1)==y)
        print("   +A offset",b,"acc all",acc.mean().round(4),"drill",acc[dr].mean().round(4),"val",acc[~dr].mean().round(4), "chosen A",(Xb.argmax(1)==0).sum())
