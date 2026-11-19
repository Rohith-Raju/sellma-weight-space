from bench.editor import Workbench
import torch, numpy as np
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b10")
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
r=wb.load_pass(1)
w=wb.weights()
main_ids=[236776, 236799, 236780, 236796]
E=w.float32(wb.embedding_name)[main_ids]
nw=w.float32("model.language_model.norm.weight")
res=r.residual; acts=r.mlp_acts
key=torch.tensor(S["shown_key"]);ch=torch.tensor(S["chosen"])
print("eps",wb.config.get("text_config",{}).get("rms_norm_eps"))
U=(1+nw)[None]*E  # 4x2560 readout dirs (before dividing by rms)
out={}
for L in range(16,34):
    W=w.float32(wb.layer_param(L)).cuda()
    g=w.float32(wb.layer_param(L,"post_feedforward_layernorm.weight")).cuda()
    raw=acts[:,L].cuda()@W.T
    mlp=(raw/raw.pow(2).mean(-1,keepdim=True).sqrt()*(1+g)).cpu()
    attn=res[:,L]-res[:,L-1]-mlp
    rms=res[:,L].pow(2).mean(-1,keepdim=True).sqrt()
    def lens(c):
        lg=(c/rms)@U.T; return lg-lg.mean(1,keepdim=True)
    la,lm,lp=lens(attn),lens(mlp),lens(res[:,L-1])
    def st(c):
        kv=c.gather(1,key[:,None])[:,0].mean().item()*4/3   # key vs rest
        cv=c.gather(1,ch[:,None])[:,0].mean().item()*4/3   # chosen vs rest
        return "key %.2f chosen %.2f bias %s"%(kv,cv,np.round(c.mean(0).numpy(),2))
    print("L%02d |attn| %.0f |mlp| %.0f |prev| %.0f\n   attn: %s\n   mlp : %s\n   prev: %s"%(L,attn.norm(dim=-1).mean(),mlp.norm(dim=-1).mean(),res[:,L-1].norm(dim=-1).mean(),st(la),st(lm),st(lp)))
    out[L]=(la,lm)
torch.save(out,"/root/sellma/editor_runs/gemma_b10/notes/decomp.pt")
