# PASS 2 (read): prefix cuts of 48 MEK prompts (24 questions x 2 orders) at candidate source positions, all layers.
from bench.editor import Workbench
import pickle, random
wb = Workbench.open("gemma_b10")
P = wb.practice_questions()
tok=wb.tokenizer
random.seed(3)
mek=[p for p in P if p.question.section=="MEK"]
sel=[]
for L in "ABCD":
    c=[p for p in mek if p.question.answer==L]; random.shuffle(c); sel+=c[:6]
texts=[];meta=[]
def find_cuts(ids):
    dec=[tok.decode([x]) for x in ids]
    n=len(ids)
    assert dec[-5:]==["<end_of_turn>","\n","<start_of_turn>","model","\n"],dec[-5:]
    cuts={}
    cuts["eot"]=n-5;cuts["nl1"]=n-4;cuts["sot"]=n-3;cuts["model"]=n-2;cuts["instr_end"]=n-6
    assert dec[n-6]==").",dec[n-6]
    # instruction letters
    assert [dec[n-13].strip(),dec[n-11].strip(),dec[n-9].strip(),dec[n-7].strip()]==list("ABCD"),dec[n-14:n-5]
    for i,L in zip([n-13,n-11,n-9,n-7],"ABCD"): cuts["ins"+L]=i
    # option labels: search backwards for pattern  (newline) L "."
    start=None
    for i in range(n-14,0,-1):
        if dec[i]=="A" and dec[i+1]=="." and dec[i-1].endswith("\n\n"): start=i;break
    assert start
    pos={"A":start}
    for L in "BCD":
        for i in range(pos[chr(ord(L)-1)]+2,n-14):
            if dec[i]==L and dec[i+1]=="." and dec[i-1]=="\n": pos[L]=i;break
        assert L in pos,(L,dec[start:n-14])
    for L in "ABCD":
        cuts["lab"+L]=pos[L]
        nxt=pos[chr(ord(L)+1)] if L!="D" else None
        if nxt: nlpos=nxt-1
        else:
            nlpos=[i for i in range(pos["D"],n-13) if dec[i]=="\n\n"][0]
        cuts["nl"+L]=nlpos; cuts["end"+L]=nlpos-1
    return cuts
for p in sel:
    for order in ["ABCD","DCBA"]:
        t=wb.render_prompt(p,order)
        ids=tok(t,add_special_tokens=False).input_ids
        cuts=find_cuts(ids)
        for name,i in cuts.items():
            pre=tok.decode(ids[:i+1])
            assert tok(pre,add_special_tokens=False).input_ids==ids[:i+1],(name,i)
            texts.append(pre); meta.append((p.id,order,name,i,len(ids)))
print(len(texts))
assert len(texts)<=1024
pickle.dump(meta,open("/root/sellma/editor_runs/gemma_b10/notes/pass2_meta.pkl","wb"))
import sys
if "--go" in sys.argv:
    r=wb.read(texts,layers=list(range(34)),top_k=5)
    print("done",wb.passes_left)
