"""Command line entry point: `uv run bench <command>`.

    bench fetch                         download booklets and answer keys into exams/raw/
    bench extract                       build exams/questions.jsonl and exams/extract_report.md
    bench split                         freeze exams/split.json (refuses to overwrite)
    bench run MODEL --split dev         run one model on one split, write runs/<id>/
    bench compare EDITED BASELINE       compare two run folders, apply config/decision.toml
    bench experiment STEP --model M     the edit experiment (see src/bench/experiment.py), resumable:
                                        select-failures | dev-subset | edits | quantized | combined |
                                        collect | all
    bench group STEP --model M          the group edit experiment (see src/bench/group_experiment.py),
                                        resumable: split | train | evaluate | quantized | collect | all
    bench editor-eval --run NAME        score an AI editor's final edit (see src/bench/editor/evaluate.py),
                                        resumable
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="download past exams and answer keys")
    fetch.add_argument("--only", nargs="*", metavar="DATE", help="only these exam dates, e.g. 2025-10-19")

    commands.add_parser("extract", help="turn the PDFs into question records")

    split = commands.add_parser("split", help="freeze the dev/test split")
    split.add_argument("--force", action="store_true", help="overwrite an existing split (avoid)")

    run = commands.add_parser("run", help="run a model on a split")
    run.add_argument("model", help="a name from config/models.toml")
    run.add_argument("--split", required=True, choices=["dev", "test", "gemma_extra"])
    run.add_argument("--limit", type=int, help="only the first N questions (for quick checks)")
    run.add_argument("--skip-damage", action="store_true", help="skip perplexity and loop checks")

    compare = commands.add_parser("compare", help="compare a run against a baseline run")
    compare.add_argument("edited", type=Path)
    compare.add_argument("baseline", type=Path)
    compare.add_argument("--control", type=Path, help="run folder of a random edit of the same size")

    experiment = commands.add_parser("experiment", help="the edit experiment, resumable")
    experiment.add_argument("step", choices=["select-failures", "dev-subset", "edits", "quantized",
                                             "combined", "collect", "all"])
    experiment.add_argument("--model", required=True, help="a full-precision model from config/models.toml")
    experiment.add_argument("--with-control-q4", action="store_true",
                            help="quantized/all: also compress and score the random controls at 4-bit")
    experiment.add_argument("--combined", action="store_true",
                            help="all: also run the combined edit (Phase 4)")

    group = commands.add_parser("group", help="the group edit experiment (src/bench/group_experiment.py), resumable")
    group.add_argument("step", choices=["split", "train", "evaluate", "quantized", "collect", "all"])
    group.add_argument("--model", required=True, help="a full-precision model from config/models.toml")
    group.add_argument("--with-control-q4", action="store_true",
                       help="quantized/all: also compress and score the random controls at 4-bit")
    group.add_argument("--max-steps", type=int, help="train/all: most training steps (default: bench.group_edit's)")

    editor_eval = commands.add_parser("editor-eval", help="score an AI editor's weight edit "
                                                          "(src/bench/editor/evaluate.py), resumable")
    editor_eval.add_argument("--model", default="gemma-4b-full", help="the full-precision model that was edited")
    editor_eval.add_argument("--run", help="the editor run: editor_runs/<run>/final_edit.safetensors")
    editor_eval.add_argument("--with-control-q4", action="store_true",
                             help="also compress and score the random control at 4-bit")

    args = parser.parse_args()

    # Imports are inside the branches so `bench fetch` works without GPU libraries installed.
    if args.command == "fetch":
        from bench.fetch import fetch_all
        fetch_all(only=args.only)
    elif args.command == "extract":
        from bench.extract import extract_all
        extract_all()
    elif args.command == "split":
        from bench.split import freeze_split
        try:
            freeze_split(force=args.force)
        except FileExistsError as error:
            raise SystemExit(f"Not overwriting: {error}") from None
    elif args.command == "run":
        from bench.run import run_model
        run_dir = run_model(args.model, args.split, limit=args.limit, skip_damage=args.skip_damage)
        print(f"Run written to {run_dir}")
    elif args.command == "compare":
        from bench.report import compare_runs
        compare_runs(args.edited, args.baseline, control=args.control)
    elif args.command == "experiment":
        from bench.experiment import run_step
        run_step(args.step, args.model, with_combined=args.combined, with_control_q4=args.with_control_q4)
    elif args.command == "group":
        from bench.group_experiment import run_step as run_group_step
        run_group_step(args.step, args.model, with_control_q4=args.with_control_q4, max_steps=args.max_steps)
    elif args.command == "editor-eval":
        from bench.editor.evaluate import evaluate_run
        if args.run is None:
            parser.error("editor-eval needs --run NAME")
        print(f"Results written to {evaluate_run(args.model, args.run, with_control_q4=args.with_control_q4)}")


if __name__ == "__main__":
    main()
