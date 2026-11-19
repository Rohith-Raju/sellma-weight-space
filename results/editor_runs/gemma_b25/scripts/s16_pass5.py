import sys; sys.path.insert(0,"editor_runs/gemma_b25/scripts")
from late_lib import *
cfg=eval(sys.argv[1])
e=late_edit(cfg["layers"],cfg["k"],cfg.get("frac",1.0))
items=evalset(*cfg.get("set",(290,222,0)))
orders=cfg.get("orders")
ans=wb.answer(items,edit=e,options_orders=[orders]*len(items) if orders else None)
if orders:
    print("reordered",orders,"acc %.3f"%np.mean([a.is_correct for a in ans]),"chosen",sorted(Counter(a.chosen_letter for a in ans).items()))
    for s in ["MEK","LAS","ORD"]:
        x=[a for a in ans if a.section==s]
        if x: print(s,len(x),"acc %.3f"%np.mean([a.is_correct for a in x]),"A-key acc %.2f"%np.mean([a.is_correct for a in x if a.correct_letter=="A"]),sorted(Counter(a.chosen_letter for a in x).items()))
else: report(items,ans,str(cfg))
