import sys; sys.path.insert(0,"editor_runs/gemma_b50/scripts")
from evalsets import *
from s20_ktest import k_edit
t1=T1(); ids={p.id for p in t1}; t2=[p for p in P if p.id not in ids]
print(len(t2))
ans=wb.answer(t2,edit=k_edit(3.5)); report(t2,ans,f"pass{wb.passes_used} K-offset gamma 3.5 on T2 (rest of practice)")
ans=wb.answer(t1,edit=k_edit(5.0)); report(t1,ans,f"pass{wb.passes_used} K-offset gamma 5.0 on T1")
