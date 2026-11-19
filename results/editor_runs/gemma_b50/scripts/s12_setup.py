meta=json.load(open(RUN+"notes/prefix_meta.json"))
names=["let0","end0","nl0","let1","end1","nl1","let2","end2","nl2","let3","end3","nl3","t_paren","t_eot","t_nl","t_sot","t_model"]
Z=torch.load(RUN+"notes/prefix_res.pt"); X=Z["X"]; key=Z["key"]; cho=Z["cho"]; qids=Z["qids"]
ix={(q,n):i for i,(q,n) in enumerate(meta)}
ntok=torch.cat([torch.as_tensor(wb.load_pass(i).n_tokens) for i in (2,3,4)])
POS={n:torch.stack([ntok[ix[(q,n)]] for q in qids])-1 for n in names}
r1=wb.load_pass(1); m1=json.load(open(RUN+"notes/pass1_meta.json"))
i1={q:i for i,(q,o) in enumerate(m1) if o is None}
F=r1.residual[[i1[q] for q in qids]]
Fpos=torch.as_tensor(r1.n_tokens)[[i1[q] for q in qids]]-1
w=wb.weights(); eps=1e-6
def rms(x,g): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*(1+g)
nw=w.float32("model.language_model.norm.weight"); E=wb.unembedding_rows([236776, 236799, 236780, 236796])
H=8; KV=4; hd=256
def rope(x,pos,base):
    inv=1.0/(base**(torch.arange(0,hd,2).float()/hd))
    fr=pos[...,None].float()*inv; emb=torch.cat([fr,fr],-1); cos=emb.cos(); sin=emb.sin()
    x1,x2=x[...,:hd//2],x[...,hd//2:]; rot=torch.cat([-x2,x1],-1)
    return x*cos+rot*sin
def attn_layer(Lr, Fd=None, Fp=None, srcs=None):
    pre=f"model.language_model.layers.{Lr}."
    Wq=w.float32(pre+"self_attn.q_proj.weight"); Wk=w.float32(pre+"self_attn.k_proj.weight"); Wv=w.float32(pre+"self_attn.v_proj.weight"); Wo=w.float32(pre+"self_attn.o_proj.weight")
    gq=w.float32(pre+"self_attn.q_norm.weight"); gk=w.float32(pre+"self_attn.k_norm.weight"); gi=w.float32(pre+"input_layernorm.weight"); gpa=w.float32(pre+"post_attention_layernorm.weight")
    base=1e6 if (Lr+1)%6==0 else 1e4
    Fd_=F[:,Lr-1] if Fd is None else Fd; Fp_=Fpos if Fp is None else Fp
    xd=rms(Fd_,gi); q=(xd@Wq.T).view(-1,H,hd); q=rms(q,gq); q=rope(q,Fp_[:,None].expand(-1,H),base)
    S=[];V=[]
    for n in names+["self"]:
        xs=rms(X[n][:,Lr-1],gi) if n!="self" else xd
        ps=POS[n] if n!="self" else Fp_
        k=(xs@Wk.T).view(-1,KV,hd); k=rms(k,gk); k=rope(k,ps[:,None].expand(-1,KV),base)
        k=k.repeat_interleave(H//KV,dim=1)
        S.append((q*k).sum(-1)*(256**-0.5))
        V.append((xs@Wv.T).view(-1,KV,hd).repeat_interleave(H//KV,dim=1))
    return torch.stack(S,-1),V,q,xd,{"Wq":Wq,"Wk":Wk,"Wv":Wv,"Wo":Wo,"gq":gq,"gk":gk,"gi":gi,"gpa":gpa}
