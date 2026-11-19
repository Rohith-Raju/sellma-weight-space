import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from evalsets import *
D=torch.load(RUN+"notes/head3_dual.pt"); md=D["md"]; Ecn=D["Ecn"]; target=D["target"]
name="model.language_model.layers.23.self_attn.o_proj.weight"
def ov_edit(mult):
    return {name:{"cols":list(range(3*256,4*256)),"values":mult*target*torch.outer(Ecn[0],md)}}
if __name__=="__main__":
    items=T1()
    for mult in [float(x) for x in sys.argv[1:]]:
        e=ov_edit(mult); sz=wb.edit_size(e); print(mult,{k:sz[k] for k in ["weights_touched","weights_changed","share_of_model"]}, sz["tensors"][name]["relative_norm"])
        ans=wb.answer(items,edit=e); report(items,ans,f"pass{wb.passes_used} OV-gain L23h3 mult {mult}")
