import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
tok=wb.tokenizer
mek=[p for p in P if p.question.section=="MEK"]
for p in mek[:2]:
    t=wb.render_prompt(p); ids=tok(t,add_special_tokens=False)["input_ids"]
    print([tok.decode([i]) for i in ids][-75:])
o=[p for p in P if p.question.section=="ORD"][0]
t=wb.render_prompt(o); ids=tok(t,add_special_tokens=False)["input_ids"]; print([tok.decode([i]) for i in ids])
