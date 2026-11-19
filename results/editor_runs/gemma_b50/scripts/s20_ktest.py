import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from evalsets import *
D=torch.load(RUN+"notes/kedit_dirs.pt"); s=D["s"]; ah=D["ah"]
name="model.language_model.layers.23.self_attn.k_proj.weight"
def k_edit(gamma):
    return {name:{"rows":list(range(256,512)),"values":gamma*torch.outer(ah,s)}}
if __name__=="__main__":
    items=T1()
    for gamma in [float(x) for x in sys.argv[1:]]:
        e=k_edit(gamma); sz=wb.edit_size(e); print(gamma,{k:sz[k] for k in ["weights_touched","weights_changed","share_of_model"]}, sz["tensors"][name]["relative_norm"])
        ans=wb.answer(items,edit=e); report(items,ans,f"pass{wb.passes_used} K-offset L23kv1 gamma {gamma}")
