import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from common import *
from collections import Counter
r=wb.load_pass(1); meta=json.load(open(RUN+"notes/pass1_meta.json"))
lt=letter_token_sets(); print({k:len(v) for k,v in lt.items()})
tok=wb.tokenizer
print([ (tok.decode([i]),i) for i in lt["A"]][:12])
print("top tokens ex:", r.top_tokens[0][:6], r.top_probs[0][:6])
# letter probs from top-k
def lp(i):
    d={l:0.0 for l in "ABCD"}
    for t,pr in zip(r.top_ids[i].tolist(), r.top_probs[i].tolist()):
        for l in "ABCD":
            if t in lt[l]: d[l]+=pr
    return d
N=len(meta)
key=[];ch=[];sec=[]
for i,(qid,order) in enumerate(meta):
    p=byid[qid]; d=lp(i); key.append(shown_key(p,order)); ch.append(max(d,key=d.get)); sec.append(p.question.section)
key=np.array(key);ch=np.array(ch);sec=np.array(sec); rot=np.array([m[1] is not None for m in meta])
# check vs baseline
bl=np.array([byid[m[0]].baseline.chosen_letter for m in meta])
print("orig agree w baseline:", (ch[~rot]==bl[~rot]).mean())
for s in ["MEK","LAS"]:
    for R in [False,True]:
        m=(sec==s)&(rot==R)
        print(s,"rot" if R else "orig","n",m.sum(),"acc",(ch[m]==key[m]).mean().round(3),"chosen",sorted(Counter(ch[m]).items()),"acc by shown key",{l:((ch[m&(key==l)]==l).mean().round(3),int((m&(key==l)).sum())) for l in "ABCD"})
# paired: same question orig vs rot
idx={ (m[0],m[1]):i for i,m in enumerate(meta)}
for s in ["MEK","LAS"]:
    tab=Counter(); tabk={}
    for (qid,o),i in idx.items():
        if o is None or byid[qid].question.section!=s: continue
        j=idx[(qid,None)]
        a=ch[j]==key[j]; b=ch[i]==key[i]
        tab[(bool(a),bool(b))]+=1
        tabk.setdefault(key[j]+"->"+key[i],Counter())[(bool(a),bool(b))]+=1
        # same content chosen?
        orig_choice=ch[j]; rot_choice_orig="BCDA"[L.index(ch[i])]
        tab["same_content" if orig_choice==rot_choice_orig else "diff_content"]+=1
    print(s,dict(tab)); 
    for k,v in sorted(tabk.items()): print("   ",k,dict(v))
np.savez(RUN+"notes/pass1_summary.npz",key=key,ch=ch,sec=sec,rot=rot)
