# PASS 1 (read): all MEK+LAS prompts original, plus reversed option order; all layers.
from bench.editor import Workbench
import numpy as np, torch, json, random
wb = Workbench.open("gemma_b25")
P = wb.practice_questions()
mek=[p for p in P if p.question.section=="MEK"]; las=[p for p in P if p.question.section=="LAS"]
random.seed(0)
lasr=[]
for l in "ABCD":
    c=[p for p in las if p.question.answer==l]; random.shuffle(c); lasr+=c[:39]
lasr=lasr[:154]
meta=[]; texts=[]
for p in mek+las:
    meta.append((p.id,"ABCD")); texts.append(wb.render_prompt(p))
for p in mek+lasr:
    meta.append((p.id,"DCBA")); texts.append(wb.render_prompt(p,"DCBA"))
print(len(texts))
json.dump(meta,open("editor_runs/gemma_b25/notes/read1_meta.json","w"))
r=wb.read(texts,layers=list(range(34)),top_k=20)
print(r.residual.shape, r.mlp_acts.shape, wb.passes_left)
