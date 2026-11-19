from bench.editor import Workbench
import torch, pickle, numpy as np
from collections import Counter, defaultdict
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b10")
P = {p.id:p for p in wb.practice_questions()}
meta=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","rb"))
r=wb.load_pass(1)
tok=wb.tokenizer
LET="ABCD"
main_ids=[wb.token_ids(L)[0] for L in LET]
print("main ids",main_ids,[tok.decode([i]) for i in main_ids])
# letter probs from top tokens
N=len(meta)
lp=np.zeros((N,4))
for i in range(N):
    for t,pr in zip(r.top_ids[i].tolist(),r.top_probs[i].tolist()):
        s=tok.decode([t]).strip()
        if s in LET and len(s)==1: lp[i,LET.index(s)]+=pr
print("example top tokens",r.top_tokens[0][:8],r.top_probs[0][:8])
shown_key=[];chosen=[];sec=[];order=[]
for i,(qid,o) in enumerate(meta):
    p=P[qid]; k=p.question.answer
    sk=o.index(k)  # shown position of key
    shown_key.append(sk); chosen.append(int(lp[i].argmax())); sec.append(p.question.section); order.append(o)
shown_key=np.array(shown_key);chosen=np.array(chosen);sec=np.array(sec);order=np.array(order)
corr=chosen==shown_key
# sanity vs baseline
bl=np.array([P[q].baseline.is_correct for q,o in meta])
print("agreement with baseline on original order",(corr[order=="ABCD"]==bl[order=="ABCD"]).mean())
for s in ["MEK","LAS"]:
    for o in ["ABCD","DCBA"]:
        m=(sec==s)&(order==o)
        print(s,o,"acc %.3f"%corr[m].mean(),"by shown key pos",[round(corr[m&(shown_key==k)].mean(),3) for k in range(4)],"chosen dist",np.bincount(chosen[m],minlength=4))
    # consistency: same original option chosen in both orders
    a=chosen[(sec==s)&(order=="ABCD")]; b=3-chosen[(sec==s)&(order=="DCBA")]
    ca=corr[(sec==s)&(order=="ABCD")]; cb=corr[(sec==s)&(order=="DCBA")]
    print(s,"same option chosen both orders %.3f"%(a==b).mean(),"both correct %.3f either %.3f"%((ca&cb).mean(),(ca|cb).mean()))
    # when inconsistent, which shown positions get chosen
    inc=a!=b
    print("   inconsistent: chosen shown pos (orig order)",np.bincount(a[inc],minlength=4),"(reversed)",np.bincount(3-b[inc],minlength=4))
# logit lens
w=wb.weights()
E=w.float32(wb.embedding_name)[main_ids]  # 4x2560
nw=w.float32("model.language_model.norm.weight")
print("norm weight stats",nw.mean().item(),nw.std().item())
res=r.residual  # N,34,2560
def lens(x,plus1):
    xn=x/ x.pow(2).mean(-1,keepdim=True).sqrt()
    g=(1+nw) if plus1 else nw
    return (xn*g)@E.T
for plus1 in [True,False]:
    lg=lens(res[:,33],plus1)
    true=torch.tensor(np.log(np.maximum(lp,1e-30)))
    d1=(lg-lg.mean(1,keepdim=True)); d2=true-true.mean(1,keepdim=True)
    ok=(lp.min(1)>0)
    print("plus1",plus1,"lens argmax agree",(lg.argmax(1).numpy()==chosen).mean(),"centered diff err",(d1-d2)[ok].abs().mean().item(), "n ok",ok.sum())
torch.save({"lp":lp,"shown_key":shown_key,"chosen":chosen,"sec":sec.tolist(),"order":order.tolist()},"/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt")
