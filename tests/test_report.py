"""Tests for run summaries and run comparisons, on small synthetic run folders (offline)."""

import json
from pathlib import Path

import numpy as np
import pytest

from bench.records import Answer, write_jsonl
from bench.report import compare_runs, summarize_run

SECTION_SIZES = {"ORD": 20, "LAS": 10, "MEK": 10}
DATES = ["2025-10-19", "2026-04-18"]


def make_answer(question_id: str, section: str, prob_correct: float, prob_on_letters: float = 0.9) -> Answer:
    is_correct = prob_correct > 0.5
    return Answer(
        question_id=question_id,
        section=section,
        correct_letter="A",
        chosen_letter="A" if is_correct else "B",
        is_correct=is_correct,
        letter_probs={"A": prob_correct * prob_on_letters, "B": (1 - prob_correct) * prob_on_letters},
        prob_on_letters=prob_on_letters,
        prob_correct=prob_correct,
    )


def question_ids() -> list[tuple[str, str]]:
    """(question id, section) for 40 questions across two exam dates."""
    ids = []
    for section, count in SECTION_SIZES.items():
        for number in range(1, count + 1):
            date = DATES[number % 2]
            ids.append((f"{date}/p3/{section}/{number}", section))
    return ids


def base_probabilities() -> np.ndarray:
    return np.random.default_rng(3).uniform(0.2, 0.9, size=len(question_ids()))


def damage(perplexity: float, loop_percent: float) -> dict:
    return {
        "perplexity": perplexity,
        "n_perplexity_docs": 120,
        "loop_rate_percent": loop_percent,
        "n_prompts": 50,
        "loop_rule": "test rule",
        "data_sha256": {},
    }


def write_run(root: Path, name: str, probabilities, split: str = "test", backend: str = "hf",
              damage_data: dict | None = damage(10.0, 0.0), ids=None) -> Path:
    """Write runs/<name>/ with answers.jsonl, run.json and damage.json."""
    run_dir = root / name
    run_dir.mkdir(parents=True)
    answers = []
    for (question_id, section), prob in zip(ids or question_ids(), probabilities):
        answers.append(make_answer(question_id, section, float(prob)))
    write_jsonl(run_dir / "answers.jsonl", answers)
    (run_dir / "run.json").write_text(json.dumps({"model": name, "backend": backend, "split": split}), encoding="utf-8")
    if damage_data is not None:
        (run_dir / "damage.json").write_text(json.dumps(damage_data), encoding="utf-8")
    return run_dir


@pytest.fixture
def runs(tmp_path):
    """A baseline, an edit that clearly helps, a control edit that does nothing, and the output root."""
    base = base_probabilities()
    rng = np.random.default_rng(4)
    edited_probs = base + 0.1 + rng.normal(0, 0.01, size=base.size)
    control_probs = base + rng.normal(0, 0.005, size=base.size)
    root = tmp_path / "runs"
    return {
        "baseline": write_run(root, "base", base),
        "edited": write_run(root, "edit", edited_probs),
        "control": write_run(root, "control", control_probs),
        "out": tmp_path / "comparisons",
        "root": root,
    }


# ---------------------------------------------------------------- summaries

def test_summarize_run_writes_json_and_markdown(runs):
    summary = summarize_run(runs["baseline"])
    assert summary["overall"]["n"] == 40
    assert summary["sections"]["ORD"]["n"] == 20
    assert set(summary["exam_dates"]) == set(DATES)
    assert summary["damage"]["perplexity"] == 10.0
    assert (runs["baseline"] / "summary.json").exists()
    assert "Run summary" in (runs["baseline"] / "summary.md").read_text(encoding="utf-8")


def test_summary_flags_ceiling_and_unsure_answers(tmp_path):
    ids = [("2025-10-19/p3/ORD/1", "ORD"), ("2025-10-19/p3/ORD/2", "ORD"), ("2025-10-19/p3/ORD/3", "ORD")]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    answers = [
        make_answer(ids[0][0], "ORD", 0.99, prob_on_letters=0.9),
        make_answer(ids[1][0], "ORD", 0.98, prob_on_letters=0.9),
        make_answer(ids[2][0], "ORD", 0.2, prob_on_letters=0.3),   # model wanted to say something else
    ]
    write_jsonl(run_dir / "answers.jsonl", answers)
    (run_dir / "run.json").write_text(json.dumps({"model": "m", "backend": "hf", "split": "dev"}), encoding="utf-8")

    summary = summarize_run(run_dir)
    ord_stats = summary["sections"]["ORD"]
    assert ord_stats["accuracy"] == pytest.approx(2 / 3)
    assert ord_stats["unsure_share"] == pytest.approx(1 / 3)
    assert ord_stats["ceiling"] is False
    assert summary["damage"] is None

    all_right = [make_answer("2025-10-19/p3/LAS/1", "LAS", 0.9)]
    write_jsonl(run_dir / "answers.jsonl", all_right)
    assert summarize_run(run_dir)["sections"]["LAS"]["ceiling"] is True


# ---------------------------------------------------------------- comparisons

def test_good_edit_with_clean_damage_is_an_improvement(runs):
    result = compare_runs(runs["edited"], runs["baseline"], control=runs["control"], out_root=runs["out"])
    assert result["verdict"] == "IMPROVEMENT"
    assert all(entry["passed"] for entry in result["rules"].values())
    assert result["indicative_only"] is False

    folder = runs["out"] / "edit__vs__base"
    assert json.loads((folder / "comparison.json").read_text(encoding="utf-8"))["verdict"] == "IMPROVEMENT"
    markdown = (folder / "comparison.md").read_text(encoding="utf-8")
    assert "## Verdict: IMPROVEMENT" in markdown
    assert "| 1_metric_improves | PASS |" in markdown


def test_damage_over_the_limit_fails_the_rule_and_the_verdict(runs):
    bad_damage = write_run(runs["root"], "edit_damaged", base_probabilities() + 0.1, damage_data=None)
    (bad_damage / "damage.json").write_text(json.dumps(damage(11.0, 5.0)), encoding="utf-8")  # +10% ppl, +5 pts loops

    result = compare_runs(bad_damage, runs["baseline"], control=runs["control"], out_root=runs["out"])
    assert result["verdict"] == "NO IMPROVEMENT"
    assert result["rules"]["1_metric_improves"]["passed"] is True
    assert result["rules"]["2_beats_control"]["passed"] is True
    assert result["rules"]["3a_perplexity"]["passed"] is False
    assert result["rules"]["3b_loop_rate"]["passed"] is False
    assert result["damage"]["perplexity"]["relative_change"] == pytest.approx(0.1)


def test_small_damage_within_limits_passes(runs):
    fine = write_run(runs["root"], "edit_fine", base_probabilities() + 0.1, damage_data=None)
    (fine / "damage.json").write_text(json.dumps(damage(10.3, 1.0)), encoding="utf-8")  # +3%, +1 pt
    result = compare_runs(fine, runs["baseline"], control=runs["control"], out_root=runs["out"])
    assert result["rules"]["3a_perplexity"]["passed"] is True
    assert result["rules"]["3b_loop_rate"]["passed"] is True


def test_missing_damage_is_shown_as_not_measured_and_blocks_improvement(runs):
    no_damage = write_run(runs["root"], "edit_no_damage", base_probabilities() + 0.1, damage_data=None)
    result = compare_runs(no_damage, runs["baseline"], control=runs["control"], out_root=runs["out"])
    assert result["verdict"] == "NO IMPROVEMENT"
    for name in ("3a_perplexity", "3b_loop_rate"):
        assert result["rules"][name]["passed"] is False
        assert result["rules"][name]["measured"] is False
    markdown = (runs["out"] / "edit_no_damage__vs__base" / "comparison.md").read_text(encoding="utf-8")
    assert "| 3a_perplexity | NOT MEASURED |" in markdown
    assert "| 3a_perplexity | FAIL |" not in markdown


def test_missing_control_counts_as_failed(runs):
    result = compare_runs(runs["edited"], runs["baseline"], out_root=runs["out"])
    assert result["verdict"] == "NO IMPROVEMENT"
    assert result["rules"]["2_beats_control"]["passed"] is False
    assert "no control run" in result["rules"]["2_beats_control"]["detail"]


def test_no_gain_is_not_an_improvement(runs):
    flat_noise = np.random.default_rng(9).normal(0, 0.01, size=40)  # no real change, only noise
    flat = write_run(runs["root"], "flat", base_probabilities() + flat_noise, damage_data=None)
    (flat / "damage.json").write_text(json.dumps(damage(10.0, 0.0)), encoding="utf-8")
    result = compare_runs(flat, runs["baseline"], control=runs["control"], out_root=runs["out"])
    assert result["rules"]["1_metric_improves"]["passed"] is False
    assert result["verdict"] == "NO IMPROVEMENT"


def test_split_mismatch_is_refused(runs):
    dev = write_run(runs["root"], "dev_edit", base_probabilities() + 0.1, split="dev", damage_data=damage(10.0, 0.0))
    with pytest.raises(ValueError, match="split differs"):
        compare_runs(dev, runs["baseline"], out_root=runs["out"])


def test_different_questions_are_refused(runs):
    shorter = question_ids()[:30]
    partial = write_run(runs["root"], "partial", base_probabilities()[:30], ids=shorter)
    with pytest.raises(ValueError, match="does not answer the same questions"):
        compare_runs(runs["edited"], partial, out_root=runs["out"])


def test_backend_mismatch_warns_and_fails_perplexity(runs):
    other_backend = write_run(runs["root"], "llama_base", base_probabilities(), backend="llamacpp")
    result = compare_runs(runs["edited"], other_backend, control=runs["control"], out_root=runs["out"])
    assert any("backends differ" in warning for warning in result["warnings"])
    assert result["rules"]["3a_perplexity"]["passed"] is False
    assert "not comparable" in result["rules"]["3a_perplexity"]["detail"]


def test_dev_split_gives_an_indicative_verdict(runs):
    dev_edit = write_run(runs["root"], "dev_edit2", base_probabilities() + 0.1, split="dev")
    dev_base = write_run(runs["root"], "dev_base", base_probabilities(), split="dev")
    dev_control = write_run(runs["root"], "dev_control", base_probabilities(), split="dev")
    result = compare_runs(dev_edit, dev_base, control=dev_control, out_root=runs["out"])
    assert result["indicative_only"] is True
    assert any("only indicative" in warning for warning in result["warnings"])
    assert "only indicative" in (runs["out"] / "dev_edit2__vs__dev_base" / "comparison.md").read_text(encoding="utf-8")


def test_gain_only_in_ord_is_not_an_improvement(runs):
    """ORD is not part of the headline, so an edit that only helps ORD must not pass."""
    ids = question_ids()
    base = base_probabilities()
    ord_only = np.array([prob + 0.1 if section == "ORD" else prob for prob, (_, section) in zip(base, ids)])
    ord_run = write_run(runs["root"], "ord_only", ord_only, damage_data=None)
    (ord_run / "damage.json").write_text(json.dumps(damage(10.0, 0.0)), encoding="utf-8")

    result = compare_runs(ord_run, runs["baseline"], control=runs["control"], out_root=runs["out"])
    assert result["headline_sections"] == ["LAS", "MEK"]
    assert result["headline"]["n"] == 20
    assert result["overall"]["n"] == 40
    assert result["rules"]["1_metric_improves"]["passed"] is False
    assert result["verdict"] == "NO IMPROVEMENT"
    assert "edited_minus_control_metric" in result
    assert "control_minus_edited_metric" not in result


def test_summary_has_headline_block_and_letter_shares_sum_to_one(runs):
    summary = summarize_run(runs["edited"])
    assert summary["headline"]["sections"] == ["LAS", "MEK"]
    assert summary["headline"]["n"] == 20
    for block in [summary["overall"], summary["headline"], *summary["sections"].values()]:
        assert sum(block["chosen_letter_share"].values()) == pytest.approx(1.0)
    markdown = (runs["edited"] / "summary.md").read_text(encoding="utf-8")
    assert "Headline (LAS+MEK)" in markdown
    assert "## Chosen letters" in markdown


def test_comparison_reports_letter_distribution_change(runs):
    result = compare_runs(runs["edited"], runs["baseline"], control=runs["control"], out_root=runs["out"])
    changes = result["headline"]["chosen_letter_change"]
    assert set(changes) == {"A", "B", "C", "D", "E"}
    assert sum(entry["change"] for entry in changes.values()) == pytest.approx(0.0, abs=1e-9)
    assert "## Chosen letters" in (runs["out"] / "edit__vs__base" / "comparison.md").read_text(encoding="utf-8")
