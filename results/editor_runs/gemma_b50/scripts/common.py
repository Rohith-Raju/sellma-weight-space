from bench.editor import Workbench
import torch, json, numpy as np
torch.set_grad_enabled(False)
wb = Workbench.open("gemma_b50")
P = wb.practice_questions()
byid={p.id:p for p in P}
L="ABCDE"
RUN="editor_runs/gemma_b50/"
def letter_token_sets():
    import os
    f=RUN+"notes/letter_tokens.json"
    if os.path.exists(f): return json.load(open(f))
    tok=wb.tokenizer; out={l:[] for l in L}
    for i in range(len(tok)):
        s=tok.decode([i]).strip()
        if s in out: out[s].append(i)
    json.dump(out,open(f,"w")); return out
def shown_key(p,order):
    if order is None: return p.question.answer
    return L[order.index(p.question.answer)]
