import sys; sys.path.insert(0,"editor_runs/gemma_b25/scripts")
from lib import *
rows=json.load(open("editor_runs/gemma_b25/notes/read1_rows.json"))
r1=wb.load_pass(1); r2=wb.load_pass(2)
E=wb.weights().float32(wb.embedding_name); nw=wb.weights().float32("model.language_model.norm.weight")
tid=[wb.token_ids(l)[0] for l in "ABCD"]; U=E[tid]
def lens(x):
    x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw); return x@U.T
ch=torch.tensor(["ABCD".index(q["chosen"]) for q in rows]); key=torch.tensor(["ABCD".index(q["key"]) for q in rows])
def bias_edit(L,beta,m=8,verbose=True,direction="class"):
    x=r1.residual[:,L]; a=r1.mlp_acts[:,L]; a2=r2.mlp_acts[:,L]
    mu=torch.stack([x[ch==k].mean(0) for k in range(4)])
    if direction=="class": d=mu[0]-mu[1:].mean(0)
    else:
        d=U[0]-U[1:].mean(0); d=d/d.norm()*(mu[0]-mu[1:].mean(0)).norm()
    W=wb.weights().float32(wb.layer_param(L)); g=wb.weights().float32(wb.layer_param(L,"post_feedforward_layernorm.weight"))
    raw=a@W.T; rms=raw.pow(2).mean(1).sqrt()
    # constant neurons at the answer position, quiet at option-end positions
    mean=a.mean(0); cv=a.std(0)/(mean.abs()+1e-9); leak=a2.abs().mean(0)/(mean.abs()+1e-9)
    score=mean.abs()/ (1+0*cv); score[(cv>0.15)|(leak>0.1)]=0
    top=score.topk(m).indices
    draw=beta*d/(1+g)*rms.mean()
    vals=torch.stack([draw/(m*mean[j]) for j in top],1)
    e={wb.layer_param(L):{"cols":top.tolist(),"values":vals}}
    if verbose:
        print("L%d neurons"%L,top.tolist(),"mean act",np.round(mean[top].numpy(),2),"cv",np.round(cv[top].numpy(),3),"leak",np.round(leak[top].numpy(),3),"n eligible",int((score>0).sum()))
        print(" rms raw mean %.3f cv %.3f | |d| %.0f | col delta norm / col norm %s"%(rms.mean(),rms.std()/rms.mean(),d.norm(),np.round((vals.norm(dim=0)/W[:,top].norm(dim=0)).numpy(),2)))
        # exact simulated new MLP output and lens effect
        raw2=raw+a[:,top]@vals.T
        o1=raw/raw.pow(2).mean(1,keepdim=True).sqrt()*(1+g); o2=raw2/raw2.pow(2).mean(1,keepdim=True).sqrt()*(1+g)
        x2=x+(o2-o1); l1=lens(x); l2=lens(x2)
        sec=np.array([q["sec"] for q in rows])
        for s in ["MEK","LAS"]:
            m_=torch.tensor(sec==s)
            print("  sim lens L%d %s: A-centred logit shift %.2f | acc %.3f -> %.3f | A picks %d -> %d of %d"%(L,s,((l2-l2.mean(1,keepdim=True))[m_,0]-(l1-l1.mean(1,keepdim=True))[m_,0]).mean(),
                (l1[m_].argmax(1)==key[m_]).float().mean(),(l2[m_].argmax(1)==key[m_]).float().mean(),(l1[m_].argmax(1)==0).sum(),(l2[m_].argmax(1)==0).sum(),m_.sum()))
        print(" ",{k:v for k,v in wb.edit_size(e).items() if k!="tensors"})
    return e
if __name__=="__main__":
    for L in [22,23,24]:
        for beta in [0.25,0.5,1.0]:
            bias_edit(L,beta)
