"""Does 4-bit compression keep the weight change? (editor 3's change, trial_b50_k)

Builds the edited Gemma folder, converts unedited and edited to bf16 GGUF and quantizes both to Q4_K_M
with llama.cpp b8680 (the same build and settings as the GPU runs), then compares the stored numbers:
  intended change  D  = bf16(edited) - bf16(unedited)        (in GGUF layout)
  surviving change Dq = deq(q4 edited) - deq(q4 unedited)
for every tensor (to confirm only the edited one differs) and in detail for the edited tensor.

  Runs on the GPU server (/root/sellma):  uv run --with gguf python -I analysis/q4_weight_check.py
"""
import hashlib, json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bench.editor.workbench import load_edit, save_edited_model   # noqa: E402

SRC = ROOT / "models/gemma-3-4b-it-official"
DATA = ROOT / "gpu_results" if (ROOT / "gpu_results").exists() else ROOT   # Mac copy vs the server itself
EDIT = DATA / "editor_runs/trial_b50_k/final_edit.safetensors"
CONTROL = DATA / "experiments/editor/trial_b50_k/control/control_edit.safetensors"   # random edit, same entries and size
# Fresh clone: the two edit files are only in the saved copy results/ (OUT below stays under DATA).
if not EDIT.exists():
    EDIT = ROOT / "results/editor_runs/trial_b50_k/final_edit.safetensors"
if not CONTROL.exists():
    CONTROL = ROOT / "results/experiments/editor/trial_b50_k/control/control_edit.safetensors"
WORK = ROOT / "work/q4check"
CONVERT = Path("/root/llama.cpp/convert_hf_to_gguf.py")       # llama.cpp b8680, same as bench.compress
CONVERT_PYTHON = "/root/llama-venv/bin/python"
QUANTIZE = "/root/llama.cpp/build/bin/llama-quantize"
QUANTIZE_THREADS = "32"                                          # as bench.compress
GPU_Q4_SHA = "fe8598adf4"   # prefix of the unedited Q4_K_M file made on the GPU machine
OUT = DATA / "experiments/analysis/reports/q4_weight_check.txt"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""): h.update(chunk)
    return h.hexdigest()

def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True); subprocess.run(list(map(str, cmd)), check=True)

def build():
    WORK.mkdir(parents=True, exist_ok=True)
    edited_dir = WORK / "edited_hf"
    if not edited_dir.exists():
        save_edited_model(SRC, load_edit(EDIT), edited_dir)
    control_dir = WORK / "control_hf"
    if not control_dir.exists() and CONTROL.exists():
        save_edited_model(SRC, load_edit(CONTROL), control_dir)
    for name, hf in (("unedited", SRC), ("edited", edited_dir), ("control", control_dir)):
        if not hf.exists():
            continue
        bf16, q4 = WORK / f"{name}-bf16.gguf", WORK / f"{name}-Q4_K_M.gguf"
        if not bf16.exists():
            run([CONVERT_PYTHON, CONVERT, hf, "--outfile", bf16, "--outtype", "bf16"])
        if not q4.exists():
            run([QUANTIZE, bf16, q4, "Q4_K_M", QUANTIZE_THREADS])

def tensors(path):
    from gguf import GGUFReader
    from gguf.quants import dequantize
    r = GGUFReader(str(path))
    return {t.name: (t.tensor_type.name, t) for t in r.tensors}, dequantize

def main():
    build()
    lines = []
    q4u_sha = sha256(WORK / "unedited-Q4_K_M.gguf"); q4e_sha = sha256(WORK / "edited-Q4_K_M.gguf")
    lines.append(f"unedited Q4_K_M sha256 {q4u_sha[:16]} (GPU-made unedited file starts {GPU_Q4_SHA}: "
                 f"{'IDENTICAL' if q4u_sha.startswith(GPU_Q4_SHA) else 'different'})")
    lines.append(f"edited   Q4_K_M sha256 {q4e_sha[:16]}")
    bu, dq = tensors(WORK / "unedited-bf16.gguf"); qu, _ = tensors(WORK / "unedited-Q4_K_M.gguf")
    for label in ("edited", "control"):
        if not (WORK / f"{label}-Q4_K_M.gguf").exists():
            continue
        be, _ = tensors(WORK / f"{label}-bf16.gguf"); qe, _ = tensors(WORK / f"{label}-Q4_K_M.gguf")
        diff_bf16 = [n for n in bu if bytes(bu[n][1].data) != bytes(be[n][1].data)]
        diff_q4 = [n for n in qu if bytes(qu[n][1].data) != bytes(qe[n][1].data)]
        lines.append(f"=== {label}: tensors differing from unedited, bf16 GGUF: {diff_bf16}; Q4_K_M GGUF: {diff_q4}")
        for name in diff_bf16:
            wu = np.asarray(dq(bu[name][1].data, bu[name][1].tensor_type), dtype=np.float64)
            we = np.asarray(dq(be[name][1].data, be[name][1].tensor_type), dtype=np.float64)
            qwu = np.asarray(dq(qu[name][1].data, qu[name][1].tensor_type), dtype=np.float64)
            qwe = np.asarray(dq(qe[name][1].data, qe[name][1].tensor_type), dtype=np.float64)
            D2, Dq2 = we - wu, qwe - qwu
            rows = np.nonzero(np.abs(D2).sum(axis=1))[0]
            in_rows = np.zeros_like(D2, dtype=bool); in_rows[rows] = True
            D, Dq = D2.ravel(), Dq2.ravel(); touched = D != 0
            proj = float(D @ Dq / (D @ D)); noise = Dq - proj * D
            cos = float(D @ Dq / (np.linalg.norm(D) * np.linalg.norm(Dq)))
            lines += [
                f"--- {name}: 4-bit type {qu[name][0]}, shape {wu.shape}; rows with the intended change {len(rows)} ({rows.min()}-{rows.max()})",
                f"entries changed in bf16: {int(touched.sum()):,}; entries changed after 4-bit: {int((Dq != 0).sum()):,} "
                f"(in the edited rows {int(((Dq != 0) & in_rows.ravel()).sum()):,}, in other rows {int(((Dq != 0) & ~in_rows.ravel()).sum()):,})",
                f"of the bf16-changed entries, still different after 4-bit: {int(((Dq != 0) & touched).sum()):,} ({100*((Dq != 0) & touched).sum()/touched.sum():.1f}%)",
                f"projection of the 4-bit change onto the intended change (D.Dq/D.D): {proj:.4f}",
                f"rounding noise left over, relative to the intended change: {np.linalg.norm(noise) / np.linalg.norm(D):.3f}; cos(D, Dq) {cos:.3f}",
                f"intended change relative to the weights in the edited rows: {np.linalg.norm(D) / np.linalg.norm(wu[rows].ravel()):.4f}; "
                f"4-bit rounding error of those weights: {np.linalg.norm((qwu - wu)[rows].ravel()) / np.linalg.norm(wu[rows].ravel()):.4f}",
            ]
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
