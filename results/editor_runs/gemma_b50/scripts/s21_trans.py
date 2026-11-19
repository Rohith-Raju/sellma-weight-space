import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from evalsets import *
from collections import Counter
items=T1()
for pn in [7,8,9]:
    ans=wb.load_pass(pn)
    for sec in ["MEK","LAS"]:
        c=Counter()
        for p,a in zip(items,ans):
            if p.question.section==sec and a.chosen_letter!=p.baseline.chosen_letter:
                c[(p.baseline.chosen_letter+">"+a.chosen_letter, "fix" if a.is_correct else ("brk" if p.baseline.is_correct else "w>w"))]+=1
        print(pn,sec,sorted(c.items()))
# rotation x multi-gap from pass 1
S=np.load(RUN+"notes/pass1_summary.npz"); meta=json.load(open(RUN+"notes/pass1_meta.json"))
key=S["key"];ch=S["ch"];sec=S["sec"];rot=S["rot"]
multi=np.array([" – " in byid[q].question.options["A"] for q,o in meta])
for R in [False,True]:
    for mu in [False,True]:
        m=(sec=="MEK")&(rot==R)&(multi==mu)
        print("MEK rot" if R else "MEK orig","multi" if mu else "single",{l:(round(float((ch[m&(key==l)]==l).mean()),2),int((m&(key==l)).sum())) for l in "ABCD"})
