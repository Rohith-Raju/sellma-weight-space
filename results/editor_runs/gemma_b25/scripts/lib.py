from bench.editor import Workbench
import numpy as np, torch, json, random
from collections import Counter
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b25")
PR = wb.practice_questions()
T=torch.load("editor_runs/gemma_b25/notes/neuron_t_mek.pt")
meta2=json.load(open("editor_runs/gemma_b25/notes/read2_meta.json")); used=set(m[0] for m in meta2)
def scale_edit(layers,k,gamma,sign=None,tt=None):
    tt=tt or T
    e={}
    for L in layers:
        W=wb.weights().float32(wb.layer_param(L)); t=tt[L].clone()
        if sign=="pos": t[t<0]=0
        if sign=="neg": t[t>0]=0
        top=t.abs().topk(k).indices
        e[wb.layer_param(L)]={"cols":top.tolist(),"values":gamma*W[:,top]}
    return e
def evalset(n_mek=290,n_las=222,n_ord=0,seed=0):
    rnd=random.Random(seed)
    out=[]
    for s,n in [("MEK",n_mek),("LAS",n_las),("ORD",n_ord)]:
        it=[p for p in PR if p.question.section==s]; rnd.shuffle(it); out+=it[:n]
    return out
def report(items,ans,tag=""):
    for s in ["MEK","LAS","ORD"]:
        idx=[i for i,p in enumerate(items) if p.question.section==s]
        if not idx: continue
        b=np.array([items[i].baseline.is_correct for i in idx]); a=np.array([ans[i].is_correct for i in idx])
        ho=np.array([items[i].id not in used for i in idx])
        print(tag,s,"n=%d base %.3f -> %.3f | fixed %d broke %d | held-out(n=%d) base %.3f -> %.3f | chosen %s | acc key=A %.2f->%.2f | mean p_correct %.3f->%.3f"%(len(idx),b.mean(),a.mean(),(a&~b).sum(),(b&~a).sum(),ho.sum(),b[ho].mean(),a[ho].mean(),
          sorted(Counter(ans[i].chosen_letter for i in idx).items()),
          np.mean([items[i].baseline.is_correct for i in idx if items[i].question.answer=="A"]),np.mean([ans[i].is_correct for i in idx if items[i].question.answer=="A"]),
          np.mean([items[i].baseline.prob_correct for i in idx]),np.mean([ans[i].prob_correct for i in idx])))
    b=np.array([p.baseline.is_correct for p in items]); a=np.array([x.is_correct for x in ans])
    print(tag,"ALL %.3f -> %.3f fixed %d broke %d"%(b.mean(),a.mean(),(a&~b).sum(),(b&~a).sum()), "passes left",wb.passes_left)
