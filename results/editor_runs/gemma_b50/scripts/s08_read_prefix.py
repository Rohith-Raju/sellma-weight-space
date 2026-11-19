# PASSES 2-4: read prefixes of MEK prompts cut at option positions and tail positions (unedited).
import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
import random
tok=wb.tokenizer
mek=[p for p in P if p.question.section=="MEK"]
random.seed(0)
sel=[]
for k in "ABCD":
    c=[p for p in mek if p.question.answer==k]; random.shuffle(c); sel+=c[:45]
random.shuffle(sel)
LET=[236776, 236799, 236780, 236796]
def cuts(p):
    ids=tok(wb.render_prompt(p),add_special_tokens=False)["input_ids"]
    n=len(ids)
    # find option block: sequence "\n\n" A . ... ; search from the end for the tail start "\n\nSvara"
    dec=[tok.decode([i]) for i in ids]
    tail=max(i for i in range(n) if dec[i]=="\n\n" and dec[i+1]=="S" and dec[i+2]=="vara")
    pos={}
    # letters: walk backwards
    starts=[]
    j=tail
    for li in [3,2,1,0]:
        k=max(i for i in range(j) if ids[i]==LET[li] and dec[i+1]=="." and dec[i-1] in ("\n","\n\n"))
        starts.append(k); j=k
    starts=starts[::-1]
    ends=[starts[1]-1,starts[2]-1,starts[3]-1,tail]   # newline token after each option
    for li in range(4):
        pos[f"let{li}"]=starts[li]; pos[f"end{li}"]=ends[li]-1; pos[f"nl{li}"]=ends[li]
    for name,off in [("t_paren",n-6),("t_eot",n-5),("t_nl",n-4),("t_sot",n-3),("t_model",n-2)]:
        pos[name]=off
    assert dec[n-5]=="<end_of_turn>" and dec[n-2]=="model", dec[-6:]
    return ids,pos
texts=[];meta=[]
for p in sel:
    ids,pos=cuts(p)
    for name,i in pos.items():
        texts.append(tok.decode(ids[:i+1])); meta.append((p.id,name))
        assert tok(texts[-1],add_special_tokens=False)["input_ids"]==ids[:i+1]
print(len(texts))
json.dump(meta,open(RUN+"notes/prefix_meta.json","w"))
for c in range(0,len(texts),1020):
    r=wb.read(texts[c:c+1020],layers=list(range(34)),top_k=5)
    print(c,r.residual.shape,wb.passes_left)
