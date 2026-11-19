"""Reports: one run's summary, and the comparison of an edited run against its baseline.

summarize_run  writes runs/<id>/summary.json and summary.md.
compare_runs   writes runs/comparisons/<edited>__vs__<baseline>/comparison.json and comparison.md,
               prints the markdown, and returns the result dict.

Two groups of questions are reported:

* The headline is the sections listed in config/decision.toml (headline_sections, LAS and MEK).
  Only the headline decides the verdict. ORD is mostly vocabulary (knowledge), so it is shown
  separately and does not count.
* The overall numbers cover every question, and each section is shown on its own.

The comparison applies config/decision.toml. An IMPROVEMENT needs all of these rules to pass:

1. The paired 95% CI of the change in the metric on the headline is above zero.
2. A control run is given (a random edit of the same size), and the edited run beats it:
   the paired CI of (edited - control) on the headline metric is above zero.
3. No damage: perplexity rises by at most max_perplexity_increase (a fraction), and the loop
   rate rises by at most max_loop_rate_increase (percentage points).

A rule that cannot be checked (no control run, no damage.json, or perplexity from different
backends) counts as failed, and the report says why.

The chosen-letter distribution is reported too. If an edit only pushed the model toward one
letter, the accuracy can move by chance while the model has not learned anything, so a large
shift in the distribution is worth a look.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from bench.paths import DECISION_CONFIG, RUNS
from bench.records import LETTERS, SECTIONS, Answer, load_answers
from bench.stats import accuracy, bootstrap_mean_difference, mean, wilson_interval

UNSURE_BELOW = 0.5          # prob_on_letters below this: the model wanted to say something other than a letter
CEILING_ACCURACY = 0.95     # at or above this, a section has little room left to show an improvement
TEST_SPLIT = "test"         # only the test split gives a verdict that counts; others are indicative


# ---------------------------------------------------------------- loading

def load_decision(path: Path = DECISION_CONFIG) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def read_run_info(run_dir: Path) -> dict:
    with open(run_dir / "run.json", encoding="utf-8") as f:
        info = json.load(f)
    return {key: info[key] for key in ("model", "backend", "split")}


def read_damage(run_dir: Path) -> dict | None:
    path = run_dir / "damage.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def exam_date(question_id: str) -> str:
    """'2025-10-19/p3/MEK/21' -> '2025-10-19'."""
    return question_id.split("/")[0]


def group_answers(answers: list[Answer], key) -> dict[str, list[Answer]]:
    groups: dict[str, list[Answer]] = {}
    for answer in answers:
        groups.setdefault(key(answer), []).append(answer)
    return groups


def headline_answers(answers: list[Answer], headline_sections: list[str]) -> list[Answer]:
    return [answer for answer in answers if answer.section in headline_sections]


def letter_shares(answers: list[Answer]) -> dict[str, float]:
    """Share of answers whose chosen letter is each letter. Letters never chosen get 0.0."""
    counts = {letter: 0 for letter in LETTERS}
    for answer in answers:
        counts[answer.chosen_letter] = counts.get(answer.chosen_letter, 0) + 1
    return {letter: counts[letter] / len(answers) for letter in LETTERS}


# ---------------------------------------------------------------- one run

def section_stats(answers: list[Answer]) -> dict:
    """The numbers for one group of answers (a section, an exam date, the headline, or all of them)."""
    correct = [answer.is_correct for answer in answers]
    low, high = wilson_interval(sum(correct), len(correct))
    unsure = sum(1 for answer in answers if answer.prob_on_letters < UNSURE_BELOW)
    score = accuracy(correct)
    return {
        "n": len(answers),
        "accuracy": score,
        "accuracy_ci95": [low, high],
        "mean_prob_correct": mean([answer.prob_correct for answer in answers]),
        "mean_prob_on_letters": mean([answer.prob_on_letters for answer in answers]),
        "unsure_share": unsure / len(answers),
        "ceiling": score >= CEILING_ACCURACY,
        "chosen_letter_share": letter_shares(answers),
    }


def summarize_run(run_dir: Path) -> dict:
    """Score, letter and damage summary of one run. Writes summary.json and summary.md into the run folder."""
    info = read_run_info(run_dir)
    headline_sections = load_decision()["headline_sections"]
    answers = load_answers(run_dir / "answers.jsonl")
    by_section = group_answers(answers, lambda answer: answer.section)
    by_date = group_answers(answers, lambda answer: exam_date(answer.question_id))
    headline = headline_answers(answers, headline_sections)
    summary = {
        **info,
        "n_questions": len(answers),
        "overall": section_stats(answers),
        "headline": {"sections": headline_sections, **section_stats(headline)} if headline else None,
        "sections": {s: section_stats(by_section[s]) for s in SECTIONS if s in by_section},
        "exam_dates": {d: section_stats(by_date[d]) for d in sorted(by_date)},
        "damage": read_damage(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    return summary


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_summary_md(summary: dict) -> str:
    lines = [
        f"# Run summary: {summary['model']}",
        "",
        f"Backend `{summary['backend']}`, split `{summary['split']}`, {summary['n_questions']} questions.",
        "",
        "## Exam score",
        "",
        "The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary "
        "(knowledge), so it is shown separately.",
        "",
        "| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = [("All", summary["overall"])]
    if summary["headline"] is not None:
        rows.append(("Headline (LAS+MEK)", summary["headline"]))
    rows += list(summary["sections"].items())
    for name, stats in rows:
        low, high = stats["accuracy_ci95"]
        lines.append(
            f"| {name} | {stats['n']} | {pct(stats['accuracy'])} ({pct(low)} to {pct(high)}) | "
            f"{stats['mean_prob_correct']:.3f} | {stats['mean_prob_on_letters']:.3f} | "
            f"{pct(stats['unsure_share'])} | {'yes' if stats['ceiling'] else 'no'} |"
        )

    lines += ["", "## Chosen letters", "", "Share of answers where each letter was the most likely choice.", "",
              "| Group | " + " | ".join(LETTERS) + " |", "|---|" + "---|" * len(LETTERS)]
    for name, stats in rows:
        shares = stats["chosen_letter_share"]
        lines.append(f"| {name} | " + " | ".join(pct(shares[letter]) for letter in LETTERS) + " |")

    lines += ["", "## By exam date", "", "| Exam date | n | Accuracy | Mean P(correct) | Unsure |", "|---|---|---|---|---|"]
    for date, stats in summary["exam_dates"].items():
        lines.append(f"| {date} | {stats['n']} | {pct(stats['accuracy'])} | "
                     f"{stats['mean_prob_correct']:.3f} | {pct(stats['unsure_share'])} |")
    lines += ["", "## Damage", ""]
    damage = summary["damage"]
    if damage is None:
        lines.append("No damage.json for this run (it was run with --skip-damage, or the checks did not finish).")
    else:
        lines += [
            f"* Perplexity: {damage['perplexity']:.3f} on {damage['n_perplexity_docs']} documents",
            f"* Loop rate: {damage['loop_rate_percent']:.1f}% of {damage['n_prompts']} prompts",
            f"* Loop rule: {damage['loop_rule']}",
        ]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- comparison

@dataclass
class LoadedRun:
    name: str                       # the run folder name, used in output paths
    info: dict                      # model, backend, split
    answers: dict[str, Answer]      # keyed by question id
    damage: dict | None


def load_run(run_dir: Path) -> LoadedRun:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / "answers.jsonl").exists():
        raise FileNotFoundError(f"no answers.jsonl in {run_dir}; is this a run folder?")
    by_id: dict[str, Answer] = {}
    for answer in load_answers(run_dir / "answers.jsonl"):
        if answer.question_id in by_id:
            raise ValueError(f"question {answer.question_id} appears twice in {run_dir}")
        by_id[answer.question_id] = answer
    return LoadedRun(run_dir.name, read_run_info(run_dir), by_id, read_damage(run_dir))


def check_comparable(edited: LoadedRun, others: list[LoadedRun]) -> list[str]:
    """Refuse runs on different splits or question sets. Return warnings for different backends."""
    warnings: list[str] = []
    for other in others:
        if other.info["split"] != edited.info["split"]:
            raise ValueError(f"split differs: {edited.name} is '{edited.info['split']}' "
                             f"but {other.name} is '{other.info['split']}'")
        missing = sorted(set(edited.answers) - set(other.answers))
        extra = sorted(set(other.answers) - set(edited.answers))
        if missing or extra:
            raise ValueError(f"{other.name} does not answer the same questions as {edited.name}: "
                             f"{len(missing)} missing (e.g. {missing[:3]}), {len(extra)} extra (e.g. {extra[:3]})")
        if other.info["backend"] != edited.info["backend"]:
            warnings.append(
                f"WARNING: backends differ ({edited.name}: {edited.info['backend']}, "
                f"{other.name}: {other.info['backend']}). Perplexity is only comparable within one backend, "
                "so the perplexity rule is treated as failed."
            )
    return warnings


def paired_change(edited: list[float], baseline: list[float], decision: dict) -> dict:
    change, low, high = bootstrap_mean_difference(
        edited, baseline,
        confidence=decision["confidence"], samples=decision["bootstrap_samples"],
    )
    return {"change": change, "ci": [low, high]}


def count_flips(edited: list[Answer], baseline: list[Answer]) -> dict[str, int]:
    """Questions whose correctness changed: wrong->right and right->wrong."""
    wrong_to_right = sum(1 for e, b in zip(edited, baseline) if e.is_correct and not b.is_correct)
    right_to_wrong = sum(1 for e, b in zip(edited, baseline) if b.is_correct and not e.is_correct)
    return {"wrong_to_right": wrong_to_right, "right_to_wrong": right_to_wrong}


def letter_change(edited: list[Answer], baseline: list[Answer]) -> dict[str, dict[str, float]]:
    """Per letter: share chosen by the edited run, by the baseline, and the difference."""
    e_shares = letter_shares(edited)
    b_shares = letter_shares(baseline)
    return {letter: {"edited": e_shares[letter], "baseline": b_shares[letter],
                     "change": e_shares[letter] - b_shares[letter]} for letter in LETTERS}


def compare_group(edited: LoadedRun, baseline: LoadedRun, ids: list[str], metric: str, decision: dict) -> dict:
    """Accuracy, metric and letter changes, with paired CIs, for the questions in ids (already sorted)."""
    e_answers = [edited.answers[i] for i in ids]
    b_answers = [baseline.answers[i] for i in ids]
    e_correct = [float(a.is_correct) for a in e_answers]
    b_correct = [float(a.is_correct) for a in b_answers]
    e_metric = [float(getattr(a, metric)) for a in e_answers]
    b_metric = [float(getattr(a, metric)) for a in b_answers]
    return {
        "n": len(ids),
        "accuracy": {
            "edited": mean(e_correct), "baseline": mean(b_correct),
            **paired_change(e_correct, b_correct, decision),
        },
        "metric": {
            "name": metric, "edited": mean(e_metric), "baseline": mean(b_metric),
            **paired_change(e_metric, b_metric, decision),
        },
        "flips": count_flips(e_answers, b_answers),
        "chosen_letter_change": letter_change(e_answers, b_answers),
    }


def compare_damage(edited: dict | None, baseline: dict | None) -> dict | None:
    if edited is None or baseline is None:
        return None
    return {
        "n_perplexity_docs": edited["n_perplexity_docs"],
        "n_prompts": edited["n_prompts"],
        "perplexity": {
            "edited": edited["perplexity"], "baseline": baseline["perplexity"],
            "relative_change": (edited["perplexity"] - baseline["perplexity"]) / baseline["perplexity"],
        },
        "loop_rate_percent": {
            "edited": edited["loop_rate_percent"], "baseline": baseline["loop_rate_percent"],
            "change_points": edited["loop_rate_percent"] - baseline["loop_rate_percent"],
        },
    }


def rule(passed: bool, detail: str, measured: bool = True) -> dict:
    """One decision rule. A rule that was not measured is stored as failed (the verdict stays strict)
    but marked measured=False, so the report shows NOT MEASURED instead of FAIL."""
    entry = {"passed": passed, "detail": detail}
    if not measured:
        entry["measured"] = False
    return entry


def rule_label(entry: dict) -> str:
    if entry.get("measured", True) is False:
        return "NOT MEASURED"
    return "PASS" if entry["passed"] else "FAIL"


DAMAGE_NOT_MEASURED = ("not measured: damage is only checked on the test papers, and one of these runs has "
                       "no damage.json (no IMPROVEMENT verdict without it)")


def evaluate_rules(headline: dict, edited_minus_control: dict | None, damage: dict | None,
                   backends_match: bool, decision: dict) -> dict:
    """The three decision rules from config/decision.toml. Each one is either passed or failed, with a reason."""
    metric = headline["metric"]
    low, high = metric["ci"]
    rules = {
        "1_metric_improves": rule(
            low > 0,
            f"headline change in {metric['name']} {metric['change']:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}] "
            "(must be above zero)",
        ),
    }

    if edited_minus_control is None:
        rules["2_beats_control"] = rule(False, "no control run given, so not checked (counts as failed)")
    else:
        c_low, c_high = edited_minus_control["ci"]
        rules["2_beats_control"] = rule(
            c_low > 0,
            f"headline edited minus control {edited_minus_control['change']:+.4f}, "
            f"95% CI [{c_low:+.4f}, {c_high:+.4f}] (must be above zero)",
        )

    if damage is None:
        rules["3a_perplexity"] = rule(False, DAMAGE_NOT_MEASURED, measured=False)
        rules["3b_loop_rate"] = rule(False, DAMAGE_NOT_MEASURED, measured=False)
    else:
        rel = damage["perplexity"]["relative_change"]
        limit = decision["max_perplexity_increase"]
        if not backends_match:
            rules["3a_perplexity"] = rule(False, "backends differ, perplexity not comparable (counts as failed)")
        else:
            rules["3a_perplexity"] = rule(
                rel <= limit, f"perplexity change {rel:+.2%} (limit +{limit:.0%})")
        points = damage["loop_rate_percent"]["change_points"]
        loop_limit = decision["max_loop_rate_increase"]
        rules["3b_loop_rate"] = rule(
            points <= loop_limit, f"loop rate change {points:+.1f} points (limit +{loop_limit:.1f})")
    return rules


def compare_runs(edited: Path, baseline: Path, control: Path | None = None,
                 out_root: Path | None = None) -> dict:
    """Compare an edited run with its baseline (and optionally a random-edit control). Writes and prints the report."""
    decision = load_decision()
    headline_sections = decision["headline_sections"]
    edited_run = load_run(edited)
    baseline_run = load_run(baseline)
    control_run = load_run(control) if control is not None else None

    others = [baseline_run] + ([control_run] if control_run is not None else [])
    warnings = check_comparable(edited_run, others)
    backends_match = all(other.info["backend"] == edited_run.info["backend"] for other in others)

    metric = decision["metric"]
    ids = sorted(edited_run.answers)
    headline_ids = [i for i in ids if edited_run.answers[i].section in headline_sections]
    if not headline_ids:
        raise ValueError(f"no questions in the headline sections {headline_sections}")

    headline = compare_group(edited_run, baseline_run, headline_ids, metric, decision)
    overall = compare_group(edited_run, baseline_run, ids, metric, decision)
    sections = {}
    for section in SECTIONS:
        section_ids = [i for i in ids if edited_run.answers[i].section == section]
        if section_ids:
            sections[section] = compare_group(edited_run, baseline_run, section_ids, metric, decision)

    edited_minus_control = None
    if control_run is not None:
        edited_minus_control = paired_change(
            [float(getattr(edited_run.answers[i], metric)) for i in headline_ids],
            [float(getattr(control_run.answers[i], metric)) for i in headline_ids],
            decision,
        )

    damage = compare_damage(edited_run.damage, baseline_run.damage)
    rules = evaluate_rules(headline, edited_minus_control, damage, backends_match, decision)
    passed = all(entry["passed"] for entry in rules.values())
    verdict = "IMPROVEMENT" if passed else "NO IMPROVEMENT"
    split = edited_run.info["split"]
    indicative = split != TEST_SPLIT
    if indicative:
        warnings.append(f"Split is '{split}', not '{TEST_SPLIT}'. The verdict is only indicative.")

    result = {
        "edited": edited_run.name,
        "baseline": baseline_run.name,
        "control": control_run.name if control_run else None,
        "split": split,
        "indicative_only": indicative,
        "n_questions": len(ids),
        "metric": metric,
        "headline_sections": headline_sections,
        "warnings": warnings,
        "headline": headline,
        "overall": overall,
        "sections": sections,
        "edited_minus_control_metric": edited_minus_control,
        "damage": damage,
        "rules": rules,
        "verdict": verdict,
    }

    root = out_root if out_root is not None else RUNS / "comparisons"
    out_dir = root / f"{edited_run.name}__vs__{baseline_run.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = render_comparison_md(result)
    (out_dir / "comparison.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return result


# ---------------------------------------------------------------- markdown

def fmt_ci(change: float, ci: list[float], scale: float = 1.0, decimals: int = 2) -> str:
    low, high = ci
    return f"{change * scale:+.{decimals}f} [{low * scale:+.{decimals}f}, {high * scale:+.{decimals}f}]"


def render_comparison_md(result: dict) -> str:
    headline_label = "+".join(result["headline_sections"])
    lines = [
        f"# Comparison: {result['edited']} vs {result['baseline']}",
        "",
        f"Split `{result['split']}`, {result['n_questions']} questions, metric `{result['metric']}`. "
        f"Headline: {headline_label}. Control: {result['control'] or 'none given'}.",
        "",
    ]
    lines += [f"* {warning}" for warning in result["warnings"]]
    if result["warnings"]:
        lines.append("")
    lines += [f"## Verdict: {result['verdict']}", ""]
    if result["indicative_only"]:
        lines += ["This split is not `test`, so this verdict is only indicative.", ""]

    lines += ["## Decision rules", "", "| Rule | Result | Detail |", "|---|---|---|"]
    for name, entry in result["rules"].items():
        lines.append(f"| {name} | {rule_label(entry)} | {entry['detail']} |")

    lines += ["", "## Exam score", "",
              f"Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. "
              f"The headline ({headline_label}) decides the verdict; ORD is shown for information only.", "",
              "| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | "
              "Change in metric (CI) | Wrong to right | Right to wrong |",
              "|---|---|---|---|---|---|---|---|"]
    groups = [(f"Headline ({headline_label})", result["headline"]), ("All questions", result["overall"])]
    groups += list(result["sections"].items())
    for name, group in groups:
        acc, metric = group["accuracy"], group["metric"]
        lines.append(
            f"| {name} | {group['n']} | {pct(acc['edited'])} | {pct(acc['baseline'])} | "
            f"{fmt_ci(acc['change'], acc['ci'], scale=100, decimals=2)} | "
            f"{fmt_ci(metric['change'], metric['ci'], decimals=4)} | "
            f"{group['flips']['wrong_to_right']} | {group['flips']['right_to_wrong']} |"
        )

    lines += ["", "## Chosen letters", "",
              "Change in the share of answers where each letter was the most likely choice (percentage points). "
              "A large shift toward one letter can move accuracy without any real learning.", "",
              "| Group | " + " | ".join(LETTERS) + " |", "|---|" + "---|" * len(LETTERS)]
    for name, group in groups:
        changes = group["chosen_letter_change"]
        lines.append(f"| {name} | " + " | ".join(
            f"{100 * changes[letter]['change']:+.1f} ({100 * changes[letter]['edited']:.1f}%)"
            for letter in LETTERS) + " |")

    edited_minus_control = result["edited_minus_control_metric"]
    if edited_minus_control is not None:
        lines += ["", "## Control", "",
                  f"Headline edited minus control on `{result['metric']}`: "
                  f"{fmt_ci(edited_minus_control['change'], edited_minus_control['ci'], decimals=4)}."]

    lines += ["", "## Damage", ""]
    damage = result["damage"]
    if damage is None:
        lines.append("Not measured for this comparison. Damage is only checked on the test papers. "
                     "Without it the verdict cannot be IMPROVEMENT.")
    else:
        p, loops = damage["perplexity"], damage["loop_rate_percent"]
        lines += [
            "| Measure | Baseline | Edited | Change |",
            "|---|---|---|---|",
            f"| Perplexity ({damage['n_perplexity_docs']} documents) | {p['baseline']:.3f} | {p['edited']:.3f} | "
            f"{p['relative_change']:+.2%} |",
            f"| Loop rate ({damage['n_prompts']} prompts) | {loops['baseline']:.1f}% | {loops['edited']:.1f}% | "
            f"{loops['change_points']:+.1f} points |",
        ]
    lines.append("")
    return "\n".join(lines)
