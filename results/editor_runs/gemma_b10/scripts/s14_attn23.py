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
w=wb.weights()
main_ids=[236776, 236799, 236780, 236796]
E=w.float32(wb.embedding_name)[main_ids]; nw=w.float32("model.language_model.norm.weight"); U=(1+nw)*E
LAYER=int(sys.argv[1]) if len(sys.argv)>1 else 23
prompts=defaultdict(dict)
for i,(qid,o,name,pos,n) in enumerate(meta): prompts[(qid,o)][name]=(i,pos,n)
names=["labA","endA","nlA","labB","endB","nlB","labC","endC","nlC","labD","endD","nlD","insA","insB","insC","insD","instr_end","eot","nl1","sot","model"]
for div in [1.0,8.0]:
    A=Attn(wb,LAYER,rope_div=div)
    allL=[];ch=[];ky=[];votes=[]
    for (qid,o),d in prompts.items():
        j=idx1[(qid,o)]; n=d["eot"][2]
        xq=r1.residual[j,LAYER-1][None]; q=A.q(xq,torch.tensor([n-1]))[0]
        src=torch.stack([r2.residual[d[nm][0],LAYER-1] for nm in names]); pos=torch.tensor([d[nm][1] for nm in names])
        # add self position
        src=torch.cat([src,xq]); pos=torch.cat([pos,torch.tensor([n-1])])
        lg=A.logits(q,A.k(src,pos))   # [8,22]
        allL.append(lg); ch.append(S["chosen"][j]); ky.append(S["shown_key"][j])
        ov=A.ov(src)  # [22,8,2560]
        votes.append(torch.einsum("mhr,lr->mhl",ov,U))  # letter readout (unnormalised) [22,8,4]
    allL=torch.stack(allL); ch=torch.tensor(ch); ky=torch.tensor(ky); votes=torch.stack(votes)
    print("===== layer",LAYER,"rope_div",div)
    print("mean logit per head (rows) x source:")
    print("      "+" ".join(n[:6].rjust(6) for n in names+["self"]))
    for h in range(8): print("h%d    "%h+" ".join("%6.1f"%v for v in allL[:,h].mean(0)))
    for kind,off in [("lab",0),("end",1),("nl",2)]:
        cols=[off,off+3,off+6,off+9]
        sub=allL[:,:,cols]  # [48,8,4]
        sel=sub.gather(2,ch[:,None,None].expand(-1,8,1))[:,:,0]; rest=(sub.sum(2)-sel)/3
        v=votes[:,cols]  # [48,4(src opt),8,4(letter)]
        own=torch.stack([v[:,p,:,p]-v[:,p].mean(-1) for p in range(4)],1).mean((0,1))  # [8]
        print(kind,"chosen-minus-rest logit per head:",np.round((sel-rest).mean(0).numpy(),2)," prior by pos (head-mean centered):",np.round((sub-sub.mean(2,keepdim=True)).mean(0).numpy(),1).tolist() if kind=="nl" else "", "\n    OV own-letter readout per head",np.round(own.numpy(),1))
    cols=[12,13,14,15]; sub=allL[:,:,cols]; sel=sub.gather(2,ch[:,None,None].expand(-1,8,1))[:,:,0]; rest=(sub.sum(2)-sel)/3
    v=votes[:,cols]; own=torch.stack([v[:,p,:,p]-v[:,p].mean(-1) for p in range(4)],1).mean((0,1))
    print("ins chosen-minus-rest:",np.round((sel-rest).mean(0).numpy(),2),"\n    prior by pos:",np.round((sub-sub.mean(2,keepdim=True)).mean(0).numpy(),1).tolist(),"\n    OV own-letter readout",np.round(own.numpy(),1))
