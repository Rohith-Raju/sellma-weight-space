import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
exec(open(RUN+"scripts/s12_setup.py").read())
Lr=23
S,V,q,xd,parts=attn_layer(Lr)
att=torch.softmax(S,-1)
nl=[names.index(f"nl{j}") for j in range(4)]
Wo=parts["Wo"]; gpa=parts["gpa"]
lensF=lambda o,l: (o*torch.rsqrt(F[:,l].pow(2).mean(-1,keepdim=True)+eps)*(1+nw))@E.T
for h in [1,2,3,7]:
    Wo_h=Wo[:,h*hd:(h+1)*hd]
    print("head",h,"attention to nl_j by condition:")
    a=att[:,h][:,nl]
    print("   mean att to nl_j when cho==j:",[round(a[cho==j,j].mean().item(),2) for j in range(4)]," when cho!=j:",[round(a[cho!=j,j].mean().item(),2) for j in range(4)])
    print("   mean att to nl_j when key==j:",[round(a[key==j,j].mean().item(),2) for j in range(4)]," when key!=j:",[round(a[key!=j,j].mean().item(),2) for j in range(4)])
    print("   raw score nl_j when key==j:",[round(S[key==j,h,nl[j]].mean().item(),2) for j in range(4)]," when key!=j:",[round(S[key!=j,h,nl[j]].mean().item(),2) for j in range(4)], "sd nonkey",[round(S[key!=j,h,nl[j]].std().item(),2) for j in range(4)])
    ov=torch.stack([lensF(V[nl[j]][:,h]@Wo_h.T*1.0,Lr).mean(0) for j in range(4)])
    print("   OV matrix raw (rows: src nl_j, cols: letter logit A..D; uncentered, pre post-attn-norm):\n",ov.round(decimals=2))
    ovl=torch.stack([lensF(V[names.index(f"let{j}")][:,h]@Wo_h.T,Lr).mean(0) for j in range(4)])
    print("   OV from let_j:\n",ovl.round(decimals=2))
# attention output norm is passed through post_attention_layernorm: check sizes
# reconstruct approx attn out from the 18 sources for all heads, compare direction to true (res[23]-res[22]-mlp)
