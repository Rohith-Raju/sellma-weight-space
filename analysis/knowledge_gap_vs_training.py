"""Do the questions Gemma never gets right (wrong in all 4 rotations) get solved by a model with more
training (Qwen3.5-9B)? Compares unedited Gemma's rotation results on dev with Qwen's standard dev run
(original option order only; Qwen was not rotation-tested). Also: what share of those questions does
editor 3's change fix (in any position)? Writes the summary to experiments/analysis/reports/."""
import glob, json, os
from pathlib import Path

ROOT = Path(os.environ.get("SELMA_ROOT", Path(__file__).resolve().parents[1]))
R = ROOT / "gpu_results" if (ROOT / "gpu_results").exists() else ROOT
if not (R / "experiments/analysis/rotation_proof.json").exists():   # fresh clone: read the saved copy
    R = ROOT / "results"
proof = json.load(open(R / "experiments/analysis/rotation_proof.json"))
gemma, edited = proof["unedited_1"], proof["editor3"]
qwen_run = sorted(glob.glob(str(R / "runs/*_qwen-9b-full_dev")))[-1]
qwen = {r["question_id"]: r for r in map(json.loads, open(qwen_run + "/answers.jsonl"))}
gemma_orig_run = sorted(glob.glob(str(R / "runs/*_gemma-4b-full_dev")))[-1]
gemma_orig = {r["question_id"]: r for r in map(json.loads, open(gemma_orig_run + "/answers.jsonl"))}

never = [q for q, v in gemma.items() if not any(v["ok"].values())]
always = [q for q, v in gemma.items() if all(v["ok"].values())]
depends = [q for q in gemma if q not in never and q not in always]
lines = []
def report(label, ids):
    qwen_right = sum(qwen[q]["is_correct"] for q in ids)
    gemma_right = sum(gemma_orig[q]["is_correct"] for q in ids)
    edit_fixed = sum(any(edited[q]["ok"].values()) for q in ids)
    lines.append(f"{label}: n={len(ids)} | Qwen-9B right (original order) {qwen_right} ({100*qwen_right/len(ids):.0f}%) "
                 f"| Gemma right (original order) {gemma_right} ({100*gemma_right/len(ids):.0f}%) "
                 f"| editor 3's change right in at least one position {edit_fixed}")
report("Gemma wrong in all 4 positions (does not know)", never)
report("Gemma right in some positions only (position-dependent)", depends)
report("Gemma right in all 4 positions (knows)", always)
mean_p = sum(gemma_orig[q]["prob_correct"] for q in never) / len(never)
lines.append(f"Gemma's mean probability on the correct option for the never-right questions (original order): {mean_p:.3f}")
out = ROOT / "experiments/analysis/reports" if not (ROOT / "gpu_results").exists() else R / "experiments/analysis/reports"
out.mkdir(parents=True, exist_ok=True)
(out / "knowledge_gap_vs_training.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
