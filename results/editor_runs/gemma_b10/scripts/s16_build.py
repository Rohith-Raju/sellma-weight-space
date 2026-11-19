import sys; sys.path.insert(0,"/root/sellma/editor_runs/gemma_b10/scripts")
from attnlib import Attn
from bench.editor import Workbench
import torch, numpy as np, pickle
from collections import defaultdict
wb = Workbench.open("gemma_b10")
meta=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass2_meta.pkl","rb"))
m1=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","rb"))
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
idx1={k:i for i,k in enumerate(m1)}
r1=wb.load_pass(1); r2=wb.load_pass(2)
LAYER=23
prompts=defaultdict(dict)
for i,(qid,o,name,pos,n) in enumerate(meta): prompts[(qid,o)][name]=(i,pos,n)
kinds=["lab","nl","ins"]
A=Attn(wb,LAYER)
qids=sorted({k[0] for k in prompts})
def basis(train_q,kinds_used=kinds,center_g=True):
    D=[]
    for kd in kinds_used:
        M=[]
        for L in "ABCD":
            M.append(torch.stack([A.norm_in(r2.residual[d[kd+L][0],LAYER-1]) for (q,o),d in prompts.items() if q in train_q]).mean(0))
        M=torch.stack(M); D.append(M-M.mean(0,keepdim=True))
    D=torch.cat(D)  # [12,2560]
    U,Sg,Vt=torch.linalg.svd(D,full_matrices=False)
    r=int((Sg>Sg[0]*1e-4).sum()); 
    return Vt[:r].T, Sg   # [2560,r]
def evaluate(WK,test_q,label):
    out={k:[] for k in kinds}; ch=[];ky=[]
    for (qid,o),d in prompts.items():
        if qid not in test_q: continue
        j=idx1[(qid,o)]; n=d["eot"][2]
        q=A.q(r1.residual[j,LAYER-1][None],torch.tensor([n-1]))[0]
        for kd in kinds:
            src=torch.stack([r2.residual[d[kd+L][0],LAYER-1] for L in "ABCD"]); pos=torch.tensor([d[kd+L][1] for L in "ABCD"])
            out[kd].append(A.logits(q,A.k(src,pos,WK)))
        ch.append(S["chosen"][j]); ky.append(S["shown_key"][j])
    ch=torch.tensor(ch);ky=torch.tensor(ky)
    print(label)
    for h in [3,2]:
        for kd in kinds:
            sub=torch.stack(out[kd])[:,h]; sel=sub.gather(1,ch[:,None])[:,0]; rest=(sub.sum(1)-sel)/3
            selk=sub.gather(1,ky[:,None])[:,0]; restk=(sub.sum(1)-selk)/3
            print("   h%d %s mean-by-pos %s chosen-rest %.2f key-rest %.2f argmax=key %.2f dist %s"%(h,kd,np.round(sub.mean(0).numpy(),1),(sel-rest).mean(),(selk-restk).mean(),(sub.argmax(1)==ky).float().mean(),np.bincount(sub.argmax(1).numpy(),minlength=4)))
        X=torch.logsumexp(torch.stack([torch.stack(out[k])[:,h] for k in kinds],0),0)
        print("   h%d combined argmax=key %.2f dist %s"%(h,(X.argmax(1)==ky).float().mean(),np.bincount(X.argmax(1).numpy(),minlength=4)))
WK=A.WK.clone()
def edited(Q):
    W=WK.clone(); W[256:512]=W[256:512]-(W[256:512]@Q)@Q.T; return W
if __name__=="__main__":
    trA=set(qids[::2]); trB=set(qids[1::2])
    evaluate(WK,trB,"== baseline on half B")
    Q,Sg=basis(trA); print("sing vals",np.round(Sg.numpy(),2),"rank",Q.shape[1])
    evaluate(edited(Q),trB,"== edit fit on half A, eval on half B")
    evaluate(WK,trA,"== baseline on half A")
    Q,Sg=basis(trB); evaluate(edited(Q),trA,"== edit fit on half B, eval on half A")
    # full
    Q,Sg=basis(set(qids))
    W=edited(Q)
    name=wb.layer_param(LAYER,"self_attn.k_proj.weight")
    delta=(W-WK)[256:512]
    edit={name:{"rows":list(range(256,512)),"values":delta}}
    print(wb.edit_size(edit))
    # rounded version
    Wr=(wb.weights()[name].float()+ (W-WK)).to(torch.bfloat16).float()
    evaluate(Wr,set(qids),"== full edit, bf16-rounded, all 24 q (in-sample)")
    torch.save({"Q":Q,"delta":delta},"/root/sellma/editor_runs/gemma_b10/notes/edit_v1.pt")
