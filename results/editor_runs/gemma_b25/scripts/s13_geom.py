from bench.editor import Workbench
import numpy as np, torch, json
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b25")
P = {p.id:p for p in wb.practice_questions()}
meta=json.load(open("editor_runs/gemma_b25/notes/read2_meta.json"))
r=wb.load_pass(2)
M=r.mlp_acts.view(256,4,34,10240); X=r.residual.view(256,4,34,2560)
ids=[meta[4*i][0] for i in range(256)]
key=torch.tensor(["ABCD".index(P[i].question.answer) for i in ids]); iskey=torch.nn.functional.one_hot(key,4).bool()
T=torch.load("editor_runs/gemma_b25/notes/neuron_t_mek.pt")
E=wb.weights().float32(wb.embedding_name)
dirs={}
for L in range(11,23):
    W=wb.weights().float32(wb.layer_param(L)); t=T[L]
    a=M[:,:,L]; ac=a-a.mean(0,keepdim=True); ac=ac-ac.mean(1,keepdim=True)
    dmean=ac[iskey].mean(0)-ac[~iskey].mean(0)          # per neuron act diff key-nonkey
    top=t.abs().topk(64).indices
    # written difference direction (raw MLP output diff) from all neurons and from top64
    v_all=W@dmean; v_top=W[:,top]@dmean[top]
    cols=W[:,top]*torch.sign(t[top]); cn=cols/cols.norm(dim=0,keepdim=True)
    cosm=(cn.T@cn); off=(cosm.sum()-64)/(64*63)
    full=(W@a.reshape(-1,10240).T).T; part=(W[:,top]@a.reshape(-1,10240)[:,top].T).T
    # residual key-nonkey mean diff at this layer
    x=X[:,:,L]; xc=x-x.mean(0,keepdim=True); xc=xc-xc.mean(1,keepdim=True); dres=xc[iskey].mean(0)-xc[~iskey].mean(0)
    cs=lambda p,q:float(p@q/(p.norm()*q.norm()))
    print("L%02d top64: mean pairwise cos of signed cols %.3f | |v_top|/|v_all| %.2f cos(v_top,v_all) %.2f | cos(v_all,resid diff) %.2f | top64 share of MLP raw out norm %.3f | n pos t in top %d | mean act top (key/non) %.2f/%.2f"%(
        L,off,v_top.norm()/v_all.norm(),cs(v_top,v_all),cs(v_all,dres),(part.norm(dim=1)/full.norm(dim=1)).mean(),(t[top]>0).sum(), a[iskey][:,top].abs().mean(), a[~iskey][:,top].abs().mean()))
    dirs[L]=dict(v_all=v_all,v_top=v_top,dres=dres)
    if L in (15,16,17):
        for nm,v in [("v_all",v_all),("-v_all",-v_all)]:
            lg=E@(v/v.norm()); print("    ",nm,[wb.tokenizer.decode([i]) for i in lg.topk(14).indices.tolist()])
torch.save(dirs,"editor_runs/gemma_b25/notes/dirs_mek.pt")
# cross-layer consistency of v_all
Ls=list(dirs); print(np.round([[float(dirs[a]["v_all"]@dirs[b]["v_all"]/(dirs[a]["v_all"].norm()*dirs[b]["v_all"].norm())) for b in Ls] for a in Ls],2))
