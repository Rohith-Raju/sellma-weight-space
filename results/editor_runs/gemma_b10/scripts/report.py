import sys, torch, pickle, numpy as np
from bench.editor import Workbench
wb = Workbench.open("gemma_b10")
P=wb.practice_questions(); byid={p.id:p for p in P}
m1=pickle.load(open("/root/sellma/editor_runs/gemma_b10/notes/pass1_meta.pkl","rb"))
S=torch.load("/root/sellma/editor_runs/gemma_b10/notes/pass1_summary.pt",weights_only=False)
idx1={k:i for i,k in enumerate(m1)}
ans=wb.load_pass(int(sys.argv[1]))
def base(a):
    o=a.options_order or "ABCDE"[:len(byid[a.question_id].question.options)]
    if o.startswith("ABCD"): 
        b=byid[a.question_id].baseline; return b.chosen_letter,b.is_correct
    j=idx1[(a.question_id,o)]; c="ABCD"[S["chosen"][j]]; return c,c==a.correct_letter
groups={}
for a in ans:
    o="rev" if (a.options_order or "A").startswith("D") else "orig"
    groups.setdefault((a.section,o),[]).append(a)
tot=[0,0,0]
for g,aa in sorted(groups.items()):
    b=[base(a) for a in aa]
    bc=np.array([x[1] for x in b]); ec=np.array([a.is_correct for a in aa])
    L=sorted({a.correct_letter for a in aa})
    print(g,"n",len(aa),"base %.3f edit %.3f"%(bc.mean(),ec.mean()),"gained",int((~bc&ec).sum()),"lost",int((bc&~ec).sum()),"changed answer",sum(x[0]!=a.chosen_letter for x,a in zip(b,aa)))
    print("    chosen base",[sum(x[0]==l for x in b) for l in L],"edit",[sum(a.chosen_letter==l for a in aa) for l in L],"key",[sum(a.correct_letter==l for a in aa) for l in L])
    print("    acc by key base",[round(np.mean([x[1] for x,a in zip(b,aa) if a.correct_letter==l]),2) for l in L],"edit",[round(np.mean([a.is_correct for a in aa if a.correct_letter==l]),2) for l in L])
    for paper in ["drill","validation"]:
        s=[i for i,a in enumerate(aa) if a.paper==paper]
        if s: print("    %s n %d base %.3f edit %.3f"%(paper,len(s),bc[s].mean(),ec[s].mean()))
    tot[0]+=bc.sum();tot[1]+=ec.sum();tot[2]+=len(aa)
print("TOTAL base %d edit %d of %d"%tuple(tot))
