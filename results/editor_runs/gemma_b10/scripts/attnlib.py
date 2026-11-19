import torch, math
torch.set_grad_enabled(False)
class Attn:
    def __init__(s,wb,L,rope_div=1.0,base=1e6):
        w=wb.weights(); f=lambda part: w.float32(wb.layer_param(L,part))
        s.gin=f("input_layernorm.weight"); s.WQ=f("self_attn.q_proj.weight"); s.WK=f("self_attn.k_proj.weight")
        s.WV=f("self_attn.v_proj.weight"); s.WO=f("self_attn.o_proj.weight"); s.qn=f("self_attn.q_norm.weight"); s.kn=f("self_attn.k_norm.weight")
        s.gpost=f("post_attention_layernorm.weight")
        s.inv=1.0/(base**(torch.arange(0,256,2).float()/256))/rope_div
    def norm_in(s,x): return x/x.pow(2).mean(-1,keepdim=True).sqrt()*(1+s.gin)
    def rope(s,v,pos):  # v [...,256], pos tensor broadcastable [...]
        fr=pos[...,None].float()*s.inv; emb=torch.cat([fr,fr],-1); c,sn=emb.cos(),emb.sin()
        v1,v2=v[...,:128],v[...,128:]; rot=torch.cat([-v2,v1],-1)
        return v*c+rot*sn
    def q(s,x,pos,WQ=None):  # x [n,2560] -> [n,8,256]
        q=(s.norm_in(x)@(s.WQ if WQ is None else WQ).T).view(-1,8,256)
        q=q/q.pow(2).mean(-1,keepdim=True).sqrt()*(1+s.qn)
        return s.rope(q,pos[:,None].expand(-1,8))
    def k(s,x,pos,WK=None):  # -> [n,4,256]
        k=(s.norm_in(x)@(s.WK if WK is None else WK).T).view(-1,4,256)
        k=k/k.pow(2).mean(-1,keepdim=True).sqrt()*(1+s.kn)
        return s.rope(k,pos[:,None].expand(-1,4))
    def logits(s,q,k):  # q [8,256], k [m,4,256] -> [8,m]
        kk=k.repeat_interleave(2,dim=1)  # [m,8,256]
        return torch.einsum("hd,mhd->hm",q,kk)/16.0
    def ov(s,x):  # x [m,2560] -> per head output [m,8,2560] (before post-attn norm)
        v=(s.norm_in(x)@s.WV.T).view(-1,4,256).repeat_interleave(2,dim=1)  # [m,8,256]
        WO=s.WO.view(2560,8,256)
        return torch.einsum("rhd,mhd->mhr",WO,v)
