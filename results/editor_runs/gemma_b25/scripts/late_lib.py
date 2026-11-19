import sys; sys.path.insert(0,"editor_runs/gemma_b25/scripts")
from lib import *
rows=json.load(open("editor_runs/gemma_b25/notes/read1_rows.json"))
r1=wb.load_pass(1)
E=wb.weights().float32(wb.embedding_name); nw=wb.weights().float32("model.language_model.norm.weight")
def letter_ids(l):
    return [i for i in [wb.token_ids(l)[0], wb.token_ids(" "+l)[0]] ]
tid=[wb.token_ids(l)[0] for l in "ABCD"]; U=E[tid]
def lens(x):
    x=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*(1+nw); return x@U.T
key=torch.tensor(["ABCD".index(q["key"]) for q in rows]); sec=np.array([q["sec"] for q in rows])
dU=(U[0]-U[1:].mean(0))*(1+nw)            # read-out direction for "A rather than B/C/D" in residual space (before final rms)
def late_edit(layers,k,frac=1.0,verbose=True):
    e={}; dx=torch.zeros_like(r1.residual[:,0]); info=[]
    for L in layers:
        W=wb.weights().float32(wb.layer_param(L)); g=wb.weights().float32(wb.layer_param(L,"post_feedforward_layernorm.weight"))
        a=r1.mlp_acts[:,L]
        dd=dU*(1+g); dhat=dd/dd.norm()              # direction in raw MLP-output space whose write changes the A-vs-rest logit
        along=dhat@W                                 # [10240]
        s=a.mean(0)*along                            # mean signed push per neuron at the answer position
        top=(-s).topk(k).indices                     # most consistently anti-A
        vals=-frac*torch.outer(dhat,along[top])
        e[wb.layer_param(L)]={"cols":top.tolist(),"values":vals}
        raw=a@W.T; raw2=raw+a[:,top]@vals.T
        o1=raw/raw.pow(2).mean(1,keepdim=True).sqrt()*(1+g); o2=raw2/raw2.pow(2).mean(1,keepdim=True).sqrt()*(1+g)
        dx+=o2-o1
        info.append((L,round(float(s[top].sum()/s[s<0].sum()),2),round(float((a[:,top]*along[top]).sum(1).std()/(a[:,top]*along[top]).sum(1).mean().abs()),2)))
    if verbose:
        print("layers",layers,"k",k,"frac",frac,"| per layer (share of total anti-A push removed, cv over prompts):",info)
        for Lr in [max(layers),33]:
            x=r1.residual[:,Lr]; l1=lens(x); l2=lens(x+dx)
            for s_ in ["MEK","LAS"]:
                m_=torch.tensor(sec==s_)
                print("  first-order sim at L%d %s: A-centred shift %.2f (sd %.2f) | acc %.3f -> %.3f | A picks %d -> %d of %d"%(Lr,s_,
                  ((l2-l2.mean(1,keepdim=True))[m_,0]-(l1-l1.mean(1,keepdim=True))[m_,0]).mean(),((l2-l2.mean(1,keepdim=True))[m_,0]-(l1-l1.mean(1,keepdim=True))[m_,0]).std(),
                  (l1[m_].argmax(1)==key[m_]).float().mean(),(l2[m_].argmax(1)==key[m_]).float().mean(),(l1[m_].argmax(1)==0).sum(),(l2[m_].argmax(1)==0).sum(),m_.sum()))
        print(" ",{k_:v for k_,v in wb.edit_size(e).items() if k_!="tensors"})
    return e
if __name__=="__main__":
    for layers in [[24],[24,25,26,27],[26,27,28,29],[24,25,26,27,28,29,30]]:
        for k in [16,64]:
            late_edit(layers,k)
