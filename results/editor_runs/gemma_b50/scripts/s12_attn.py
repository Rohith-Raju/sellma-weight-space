import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
c=wb.config; tc=c.get("text_config",c)
print({k:tc.get(k) for k in ["num_attention_heads","num_key_value_heads","head_dim","query_pre_attn_scalar","rope_theta","rope_local_base_freq","sliding_window","sliding_window_pattern","layer_types","rms_norm_eps","attn_logit_softcapping"]})
meta=json.load(open(RUN+"notes/prefix_meta.json"))
names=["let0","end0","nl0","let1","end1","nl1","let2","end2","nl2","let3","end3","nl3","t_paren","t_eot","t_nl","t_sot","t_model"]
Z=torch.load(RUN+"notes/prefix_res.pt"); X=Z["X"]; key=Z["key"]; cho=Z["cho"]; qids=Z["qids"]
ix={(q,n):i for i,(q,n) in enumerate(meta)}
ntok=torch.cat([torch.as_tensor(wb.load_pass(i).n_tokens) for i in (2,3,4)])
POS={n:torch.stack([ntok[ix[(q,n)]] for q in qids])-1 for n in names}   # absolute index incl BOS
r1=wb.load_pass(1); m1=json.load(open(RUN+"notes/pass1_meta.json"))
i1={q:i for i,(q,o) in enumerate(m1) if o is None}
F=r1.residual[[i1[q] for q in qids]]      # [180,34,2560] final position
Fpos=torch.as_tensor(r1.n_tokens)[[i1[q] for q in qids]]-1
print("pos check", POS["t_model"][:3], Fpos[:3])
w=wb.weights(); eps=1e-6
def rms(x,g): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*(1+g)
nw=w.float32("model.language_model.norm.weight"); E=wb.unembedding_rows([236776, 236799, 236780, 236796])
H=tc.get("num_attention_heads",8); KV=tc.get("num_key_value_heads",4); hd=tc.get("head_dim",256)
def rope(x,pos,base):
    # x [...,hd], pos [...]
    inv=1.0/(base**(torch.arange(0,hd,2).float()/hd))
    fr=pos[...,None].float()*inv; emb=torch.cat([fr,fr],-1); cos=emb.cos(); sin=emb.sin()
    x1,x2=x[...,:hd//2],x[...,hd//2:]; rot=torch.cat([-x2,x1],-1)
    return x*cos+rot*sin
res={}
for Lr in [20,21,22,23,24,25]:
    pre=f"model.language_model.layers.{Lr}."
    Wq=w.float32(pre+"self_attn.q_proj.weight"); Wk=w.float32(pre+"self_attn.k_proj.weight"); Wv=w.float32(pre+"self_attn.v_proj.weight"); Wo=w.float32(pre+"self_attn.o_proj.weight")
    gq=w.float32(pre+"self_attn.q_norm.weight"); gk=w.float32(pre+"self_attn.k_norm.weight"); gi=w.float32(pre+"input_layernorm.weight"); gpa=w.float32(pre+"post_attention_layernorm.weight")
    base=1e6 if (Lr+1)%6==0 else 1e4
    xd=rms(F[:,Lr-1],gi); q=(xd@Wq.T).view(-1,H,hd); q=rms(q,gq); q=rope(q,Fpos[:,None].expand(-1,H),base)
    S=[];V=[]
    for n in names+["self"]:
        xs=rms(X[n][:,Lr-1],gi) if n!="self" else xd
        ps=POS[n] if n!="self" else Fpos
        k=(xs@Wk.T).view(-1,KV,hd); k=rms(k,gk); k=rope(k,ps[:,None].expand(-1,KV),base)
        k=k.repeat_interleave(H//KV,dim=1)
        S.append((q*k).sum(-1)*(tc.get("query_pre_attn_scalar",256)**-0.5))
        v=(xs@Wv.T).view(-1,KV,hd).repeat_interleave(H//KV,dim=1)   # [N,H,hd]
        V.append(v)
    S=torch.stack(S,-1)   # [N,H,18]
    att=torch.softmax(S,-1)
    nl=[names.index(f"nl{j}") for j in range(4)]
    print(f"=== L{Lr} (base {base:g}) partial attention from final pos, mean over items; cols: "+" ".join(names+["self"]))
    for h in range(H):
        a=att[:,h].mean(0)
        am=S[:,h][:,nl].argmax(-1)
        # OV: what does value at nl_j write in letter space through this head
        Wo_h=Wo[:,h*hd:(h+1)*hd]
        ov=[]
        for j in range(4):
            o=V[nl[j]][:,h]@Wo_h.T    # [N,2560]
            ov.append(((o*torch.rsqrt(F[:,Lr].pow(2).mean(-1,keepdim=True))*(1+nw)*(1+gpa))@(E-E.mean(0)).T).mean(0))
        ov=torch.stack(ov)  # [src j, letter]
        print(f"h{h} att "+" ".join(f"{x:.2f}" for x in a.tolist())+f" | argmax nl==cho {(am==cho).float().mean():.2f} ==key {(am==key).float().mean():.2f} | OV diag {[round(ov[j,j].item(),2) for j in range(4)]} offdiag {round(((ov.sum()-ov.diag().sum())/12).item(),2)}")
    res[Lr]=S
torch.save(res,RUN+"notes/attn_scores.pt")
