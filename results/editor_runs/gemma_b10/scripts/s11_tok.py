from bench.editor import Workbench
import json
wb = Workbench.open("gemma_b10")
P = wb.practice_questions()
tc=wb.config.get("text_config",wb.config)
print({k:v for k,v in tc.items() if k not in ("layer_types",)})
print(tc.get("layer_types"))
tok=wb.tokenizer
it=[p for p in P if p.question.section=="MEK"][0]
t=wb.render_prompt(it)
ids=tok(t,add_special_tokens=False).input_ids
print([(i,tok.decode([x])) for i,x in enumerate(ids)][-75:])
w=wb.weights()
for part in ["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight","self_attn.o_proj.weight","self_attn.q_norm.weight","self_attn.k_norm.weight"]:
    n=wb.layer_param(23,part); print(n,w.shape(n))
