import torch
E=torch.load("/root/sellma/editor_runs/gemma_b10/notes/edit_q_v1.pt")
torch.save({"rows":E["rows"],"delta":0.5*E["delta"]},"/root/sellma/editor_runs/gemma_b10/notes/edit_q_v1_half.pt")
from bench.editor import Workbench
wb=Workbench.open("gemma_b10"); name=wb.layer_param(23,"self_attn.q_proj.weight")
s=wb.edit_size({name:{"rows":E["rows"],"values":0.5*E["delta"]}}); print(s["weights_touched"],s["weights_changed"])
