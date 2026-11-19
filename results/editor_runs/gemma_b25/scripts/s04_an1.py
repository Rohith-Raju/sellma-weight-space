from bench.editor import Workbench
import numpy as np, torch, json
from collections import Counter
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b25")
P = {p.id:p for p in wb.practice_questions()}
meta=json.load(open("editor_runs/gemma_b25/notes/read1_meta.json"))
r=wb.load_pass(1)
print(r.top_tokens[0][:8], r.top_probs[0][:8])
def letters(i):
    d=Counter()
    for t,p in zip(r.top_tokens[i],r.top_probs[i].tolist()):
        s=t.strip()
        if s in list("ABCDE"): d[s]+=p
    return d
rev={"A":"D","B":"C","C":"B","D":"A"}
rows=[]
for i,(qid,order) in enumerate(meta):
    p=P[qid]; d=letters(i); ch=max(d,key=d.get)
    key=p.question.answer if order=="ABCD" else rev[p.question.answer]
    rows.append(dict(i=i,id=qid,sec=p.question.section,order=order,chosen=ch,key=key,ok=ch==key,origkey=p.question.answer,base_ok=p.baseline.is_correct,base_ch=p.baseline.chosen_letter))
# sanity vs baseline
print("orig agree w baseline", np.mean([x["chosen"]==x["base_ch"] for x in rows if x["order"]=="ABCD"]))
for sec in ["MEK","LAS"]:
    for order in ["ABCD","DCBA"]:
        it=[x for x in rows if x["sec"]==sec and x["order"]==order]
        print(sec,order,len(it),"acc %.3f"%np.mean([x["ok"] for x in it]),"chosen",sorted(Counter(x["chosen"] for x in it).items()),
          "acc by shown key",{l:(round(np.mean([x["ok"] for x in it if x["key"]==l]),2),sum(x["key"]==l for x in it)) for l in "ABCD"})
    # consistency: same items both orders
    o={x["id"]:x for x in rows if x["sec"]==sec and x["order"]=="ABCD"}
    v=[x for x in rows if x["sec"]==sec and x["order"]=="DCBA"]
    print("  same content chosen both:",np.mean([rev[x["chosen"]]==o[x["id"]]["chosen"] for x in v]),
          " both ok",np.mean([x["ok"] and o[x["id"]]["ok"] for x in v])," either ok",np.mean([x["ok"] or o[x["id"]]["ok"] for x in v]),
          "orig acc on these",np.mean([o[x["id"]]["ok"] for x in v]))
json.dump(rows,open("editor_runs/gemma_b25/notes/read1_rows.json","w"))
# logit lens
E=wb.weights().float32(wb.embedding_name)
nw=wb.weights().float32("model.language_model.norm.weight")
print("norm cfg", {k:v for k,v in wb.config.get("text_config",wb.config).items() if "eps" in k or "norm" in k})
tid=[wb.token_ids(l)[0] for l in "ABCD"]; print(tid,[wb.tokenizer.decode([t]) for t in tid], r.top_ids[0][:5])
U=E[tid]  # 4x2560
res=r.residual  # n,34,2560
def lens(x):
    x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw)
    return x@U.T
keyidx=torch.tensor(["ABCD".index(x["key"]) for x in rows])
chidx=torch.tensor(["ABCD".index(x["chosen"]) for x in rows])
for sec in ["MEK","LAS"]:
    sel=torch.tensor([x["sec"]==sec for x in rows])
    print(sec)
    for L in range(0,34):
        lg=lens(res[sel,L])  # n,4
        lg=lg-lg.mean(1,keepdim=True)
        am=lg.argmax(1)
        print("  L%02d mean centred logit A..D %s | lens-acc %.3f agree-final %.3f | pred dist %s | margin std %.2f"%(L,np.round(lg.mean(0).numpy(),2),(am==keyidx[sel]).float().mean(),(am==chidx[sel]).float().mean(),torch.bincount(am,minlength=4).tolist(), lg.std()))
