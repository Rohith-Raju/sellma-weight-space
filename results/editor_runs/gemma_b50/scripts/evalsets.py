from common import *
def T1():
    mek=[p for p in P if p.question.section=="MEK"]; las=[p for p in P if p.question.section=="LAS"]; ord_=[p for p in P if p.question.section=="ORD"]
    return mek+las[:122]+ord_[:100]
def report(items,answers,tag=""):
    from collections import Counter
    out=[]
    for sec in ["MEK","LAS","ORD"]:
        idx=[i for i,p in enumerate(items) if p.question.section==sec]
        if not idx: continue
        b=sum(items[i].baseline.is_correct for i in idx); a=sum(answers[i].is_correct for i in idx)
        fix=sum((not items[i].baseline.is_correct) and answers[i].is_correct for i in idx); brk=sum(items[i].baseline.is_correct and (not answers[i].is_correct) for i in idx)
        ch=Counter(answers[i].chosen_letter for i in idx); kA=[i for i in idx if items[i].question.answer=="A"]
        out.append(f"{sec} n={len(idx)} base {b} -> {a} (fixed {fix}, broke {brk}) choseA {ch.get(chr(65),0)} (base {sum(items[i].baseline.chosen_letter==chr(65) for i in idx)}) keyA acc {sum(answers[i].is_correct for i in kA)}/{len(kA)} (base {sum(items[i].baseline.is_correct for i in kA)})")
    b=sum(p.baseline.is_correct for p in items); a=sum(x.is_correct for x in answers)
    s=f"[{tag}] total {b} -> {a} of {len(items)} | "+" | ".join(out)
    print(s); open(RUN+"notes/results.log","a").write(s+"\n"); return s
