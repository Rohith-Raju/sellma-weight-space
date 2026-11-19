"""Compare a fresh run of the main experiment with the saved results in results/.

The main-experiment scripts (analysis/rotation_proof.py, heldout_position.py, other_exams.py) write
under SELMA_RESULTS (default: the repository root, see analysis/_paths.py). This script reads what
they wrote there and the saved copies under results/, and prints, per file, how many answers differ.
It changes nothing.

    uv run python scripts/check_main.py            (or: make check-main)

Exact agreement is expected on the same GPU type and library versions (the saved repeats agree on
all 3,600 dev answers). On other hardware a few answers that sit within a hair of a tie may flip;
the margins in rotation_proof.json show how close each one was.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from _paths import RESULTS as NEW  # noqa: E402

SAVED = ROOT / "results"
A = "experiments/analysis"


def load(path: Path):
    return json.load(open(path, encoding="utf-8"))


def rotation_proof() -> None:
    new_path, saved_path = NEW / "logs/rotation_proof.json", SAVED / A / "rotation_proof.json"
    if not new_path.exists():
        print(f"rotation_proof: no new file at {new_path}"); return
    new, saved = load(new_path), load(saved_path)
    for cond in saved:
        if cond not in new:
            print(f"rotation_proof {cond:15} missing in new file"); continue
        ids = list(saved[cond])
        flips = sum(new[cond][q]["ok"][L] != saved[cond][q]["ok"][L] for q in ids for L in "ABCD")
        dm = max(abs(new[cond][q]["margin"][L] - saved[cond][q]["margin"][L]) for q in ids for L in "ABCD")
        print(f"rotation_proof {cond:15} {len(ids)} questions x 4 positions: {flips} answers differ; "
              f"largest margin difference {dm:.2e}")


def heldout_position() -> None:
    new_path, saved_path = NEW / A / "heldout_position.jsonl", SAVED / A / "heldout_position.jsonl"
    if not new_path.exists():
        print(f"heldout_position: no new file at {new_path}"); return
    def rows(p):
        return {(r["condition"], r["set"], r["id"]): r for r in map(json.loads, open(p, encoding="utf-8"))}
    new, saved = rows(new_path), rows(saved_path)
    fields = ("original_order_correct", "option_average_correct", "prompt_note_correct", "correct_by_position")
    diff = {f: 0 for f in fields}
    missing = 0
    for key, s in saved.items():
        n = new.get(key)
        if n is None:
            missing += 1; continue
        for f in fields:
            diff[f] += n.get(f) != s.get(f)
    print(f"heldout_position {len(saved)} rows ({missing} missing in new file); rows that differ: "
          + ", ".join(f"{f} {v}" for f, v in diff.items()))
    new_sum, saved_sum = NEW / A / "heldout_position.json", SAVED / A / "heldout_position.json"
    if new_sum.exists():
        same = load(new_sum)["summary"] == load(saved_sum)["summary"]
        print(f"heldout_position summary (heldout_position.json): {'identical' if same else 'DIFFERENT'}")


def other_exams() -> None:
    for cond in ("unedited", "editor3", "random"):
        new_path = NEW / A / "other_exams" / f"{cond}_with_choices.json"
        if not new_path.exists():
            print(f"other_exams {cond:8}: no new file at {new_path}"); continue
        new = load(new_path)["results"]
        # Correctness per position: saved by the first version of the script (other_exams_v1_first_run.py).
        saved = load(SAVED / A / "other_exams" / f"{cond}.json")["results"]
        for s in ("belebele_sv", "belebele_en", "mmlu_math"):
            ids = list(saved[s])
            flips = sum(new[s][q][L] != saved[s][q][L] for q in ids for L in "ABCD")
            print(f"other_exams {cond:8} {s:12} {len(ids)} questions x 4: {flips} answers differ")
        # Chosen letters: saved only for unedited and editor3.
        saved_choices = SAVED / A / "other_exams" / f"{cond}_with_choices.json"
        if saved_choices.exists():
            sc = load(saved_choices)["results"]
            for s in ("belebele_sv", "belebele_en", "mmlu_math"):
                k = s + "/chosen"; ids = list(sc[k])
                flips = sum(new[k][q][L] != sc[k][q][L] for q in ids for L in sc[k][q])
                print(f"other_exams {cond:8} {s:12} chosen letters: {flips} differ")


if __name__ == "__main__":
    if NEW.resolve() == SAVED.resolve():
        sys.exit("SELMA_RESULTS points at results/ itself; run the experiment into another folder first.")
    print(f"new results: {NEW}\nsaved results: {SAVED}\n")
    rotation_proof()
    heldout_position()
    other_exams()
