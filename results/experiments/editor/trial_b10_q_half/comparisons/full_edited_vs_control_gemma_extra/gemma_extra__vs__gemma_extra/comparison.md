# Comparison: gemma_extra vs gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0217, 95% CI [-0.0610, +0.0069] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 70.0% | 72.5% | -2.50 [-6.25, +0.00] | -0.0217 [-0.0610, +0.0069] | 0 | 2 |
| All questions | 120 | 75.8% | 77.5% | -1.67 [-4.17, +0.00] | -0.0153 [-0.0419, +0.0035] | 0 | 2 |
| ORD | 40 | 87.5% | 87.5% | +0.00 [+0.00, +0.00] | -0.0026 [-0.0078, -0.0000] | 0 | 0 |
| LAS | 40 | 65.0% | 67.5% | -2.50 [-7.50, +0.00] | -0.0171 [-0.0735, +0.0185] | 0 | 1 |
| MEK | 40 | 75.0% | 77.5% | -2.50 [-7.50, +0.00] | -0.0262 [-0.0825, +0.0075] | 0 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.3 (22.5%) | +1.3 (32.5%) | +1.3 (20.0%) | -3.7 (25.0%) | +0.0 (0.0%) |
| All questions | +0.8 (20.0%) | +0.8 (26.7%) | +0.8 (20.0%) | -2.5 (27.5%) | +0.0 (5.8%) |
| ORD | +0.0 (15.0%) | +0.0 (15.0%) | +0.0 (20.0%) | +0.0 (32.5%) | +0.0 (17.5%) |
| LAS | +0.0 (20.0%) | +2.5 (32.5%) | +2.5 (15.0%) | -5.0 (32.5%) | +0.0 (0.0%) |
| MEK | +2.5 (25.0%) | +0.0 (32.5%) | +0.0 (25.0%) | -2.5 (17.5%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
