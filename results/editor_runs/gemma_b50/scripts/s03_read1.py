# PASS 1: read, unedited model. 580 MEK/LAS prompts in original order + 444 rotated (BCDA). All layers.
from bench.editor import Workbench
import torch, json
wb = Workbench.open("gemma_b50")
P = wb.practice_questions()
mek=[p for p in P if p.question.section=="MEK"]; las=[p for p in P if p.question.section=="LAS"]
items=mek+las
meta=[(p.id,None) for p in items]
rot=mek+las[:154]
meta+= [(p.id,"BCDA") for p in rot]
texts=[wb.render_prompt(p) for p in items]+[wb.render_prompt(p,"BCDA") for p in rot]
assert len(texts)==1024
json.dump(meta,open("editor_runs/gemma_b50/notes/pass1_meta.json","w"))
r=wb.read(texts,layers=list(range(34)),top_k=50)
print(r.residual.shape, r.mlp_acts.shape, wb.passes_left)
