"""How much of the model each weight edit changed (reads saved edit records, no model).

"Changed" = the stored 16-bit (bfloat16) value of a weight is different after the edit. A rank-1 or low-rank
change touches every entry of the matrices it is added to, but many tiny changes round away in bfloat16.
Denominators are all weights in the checkpoint, as the editor workbench counts them:
  Gemma 3 4B it: 4,300,079,472 (includes the vision tower; from editor_runs/*/final_edit.json "model_weights")
  Qwen3.5-9B:    9,653,104,368 (sum of tensor sizes in the safetensors headers of models/Qwen3.5-9B-HF, counted on
                 the GPU machine on 2026-11-19)
Matrix sizes: Gemma layer down_proj [2560, 10240]; Qwen layer 21 down_proj [4096, 12288].
"""
import ast, glob, json
from _paths import RESULTS

GEMMA, QWEN = 4_300_079_472, 9_653_104_368
MAT = {"gemma-4b-full": 2560 * 10240, "qwen-9b-full": 4096 * 12288}
TOTAL = {"gemma-4b-full": GEMMA, "qwen-9b-full": QWEN}

print("One weight edit per wrong question (rank-1 change to one down_proj matrix), fixed cases:")
for m in MAT:
    fr = []
    for f in sorted(glob.glob(str(RESULTS / f"archive/single_question_edits/{m}/edits/*/search.json"))):
        j = json.load(open(f))
        if j.get("fixed"):
            t = [t for t in j["tried"] if t["alpha_fraction"] == j["edit"]["alpha_fraction"]][0]
            fr.append(t["entries_changed"])
    lo, hi = min(fr) * MAT[m], max(fr) * MAT[m]
    print(f"  {m}: {len(fr)} edits, {lo:,.0f} to {hi:,.0f} weights changed = "
          f"{100 * lo / TOTAL[m]:.3f}% to {100 * hi / TOTAL[m]:.3f}% of the model")

d = json.load(open(RESULTS / "experiments/group/gemma-4b-full/trained/delta.json"))
info = d["info"] if isinstance(d["info"], dict) else ast.literal_eval(d["info"])
ch = sum(L["entries_changed"] * MAT["gemma-4b-full"] for L in info["layers"])
print(f"Group edit (rank {d['rank']}, down_proj of layers {d['layers']}): {ch:,.0f} weights changed = "
      f"{100 * ch / GEMMA:.2f}% of the model ({4 * int(d['rank']) * (2560 + 10240):,} trained numbers)")

for f in sorted(glob.glob(str(RESULTS / "editor_runs/trial_*/final_edit.json"))):
    j = json.load(open(f))
    sz = j["size"] if isinstance(j["size"], dict) else ast.literal_eval(j["size"])
    print(f"AI editor trial {f.split('/')[-2]}: {sz['weights_changed']:,} weights changed = "
          f"{100 * sz['weights_changed'] / sz['model_weights']:.4f}% of the model "
          f"({sz['weights_touched']:,} touched = {100 * sz['share_of_model']:.4f}%)")
