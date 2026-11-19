import json, numpy as np
from collections import Counter
rows=json.load(open("editor_runs/gemma_b25/notes/read1_rows.json"))
rev={"A":"D","B":"C","C":"B","D":"A"}
for sec in ["MEK","LAS"]:
    o={x["id"]:x for x in rows if x["sec"]==sec and x["order"]=="ABCD"}
    v=[x for x in rows if x["sec"]==sec and x["order"]=="DCBA"]
    same_content=[rev[x["chosen"]]==o[x["id"]]["chosen"] for x in v]
    same_letter=[x["chosen"]==o[x["id"]]["chosen"] for x in v]
    print(sec,len(v),"same content %.3f same letter %.3f"%(np.mean(same_content),np.mean(same_letter)))
    print("  acc when consistent %.3f (n=%d); when inconsistent: orig %.3f rev %.3f"%(np.mean([x["ok"] for x,s in zip(v,same_content) if s]),sum(same_content),
       np.mean([o[x["id"]]["ok"] for x,s in zip(v,same_content) if not s]),np.mean([x["ok"] for x,s in zip(v,same_content) if not s])))
    # inconsistent: which positions chosen
    print("  inconsistent picks (orig pos, rev pos):",sorted(Counter((o[x["id"]]["chosen"],x["chosen"]) for x,s in zip(v,same_content) if not s).items(),key=lambda t:-t[1]))
    # confusion: shown key position -> chosen position, pooled over both orders
    for kpos in "ABCD":
        c=Counter(x["chosen"] for x in rows if x["sec"]==sec and x["key"]==kpos)
        print("   key",kpos,dict(sorted(c.items())))
