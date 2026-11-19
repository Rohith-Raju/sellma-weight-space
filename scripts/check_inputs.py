"""Check that the inputs on this machine are the ones the saved results were made with.

    uv run python scripts/check_inputs.py data      exams/questions.jsonl and exams/split.json
    uv run python scripts/check_inputs.py models    model folders and our 4-bit GGUF files
    (or: make check-data / make check-models)

The expected checksums are read from the saved files themselves (exams/split.json and the run.json
of the standard runs in results/runs/), not typed in here. Exit code 1 if anything differs.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def saved_run(model: str) -> dict:
    run = sorted(RUNS.glob(f"*_{model}_test"))[-1] / "run.json"
    return json.loads(run.read_text(encoding="utf-8"))


def check(label: str, path: Path, expected: str) -> bool:
    if not path.exists():
        print(f"MISSING  {label}: {path.relative_to(ROOT)}")
        return False
    got = sha256(path)
    ok = got == expected
    print(f"{'ok      ' if ok else 'DIFFERS '} {label}: {path.relative_to(ROOT)} {got[:12]}… (expected {expected[:12]}…)")
    return ok


def data() -> bool:
    split = json.loads((ROOT / "exams/split.json").read_text(encoding="utf-8"))
    ok = check("questions (vs split.json)", ROOT / "exams/questions.jsonl", split["questions_sha256"])
    run = saved_run("gemma-4b-full")
    ok &= check("questions (vs saved runs)", ROOT / "exams/questions.jsonl", run["questions_sha256"])
    ok &= check("split (vs saved runs)", ROOT / "exams/split.json", run["split_sha256"])
    return ok


def models() -> bool:
    ok = True
    for model in ("gemma-4b-full", "qwen-9b-full"):
        describe = saved_run(model)["describe"]
        folder = ROOT / "models" / Path(describe["path"]).name   # models/<folder> as in config/models.toml
        for name, expected in sorted(describe["model_files"].items()):
            ok &= check(model, folder / name, expected)
    for model in ("gemma-4b-q4", "qwen-9b-q4"):
        describe = saved_run(model)["describe"]
        ok &= check(model, ROOT / "models/compressed" / Path(describe["path"]).name, describe["sha256"])
    return ok


if __name__ == "__main__":
    what = sys.argv[1:] or ["data"]
    results = [{"data": data, "models": models}[w]() for w in what]
    sys.exit(0 if all(results) else 1)
