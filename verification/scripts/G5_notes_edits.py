"""G5: do the edits saved in gemma_b10/notes equal the edits the ledgered passes used? (evalpass.py takes its edit path from argv)"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import torch
from pathlib import Path
from safetensors.torch import load_file
ER = Path(_RES + "/editor_runs/gemma_b10")
name = "model.language_model.layers.23.self_attn.q_proj.weight"
for note, p in (("edit_q_v1.pt", "003_answer"), ("edit_q_v1_half.pt", "004_answer"), ("edit_q_v1_half.pt", "005_answer"), ("edit_v1.pt", None)):
    E = torch.load(ER / "notes" / note, weights_only=True)
    print(note, "keys", list(E))
    if p is None: continue
    t = load_file(str(ER / "passes" / p / "edit.safetensors"))
    rows = torch.as_tensor(E["rows"]); vals = torch.as_tensor(E["delta"]).float()
    print("   ==", p, ":", torch.equal(rows, t["rows:" + name]) and torch.equal(vals, t["values:" + name]))
