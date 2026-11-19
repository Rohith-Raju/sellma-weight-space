# Comparison: gemma_extra vs gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0283, 95% CI [-0.0682, +0.0020] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 68.8% | 72.5% | -3.75 [-8.75, +0.00] | -0.0283 [-0.0682, +0.0020] | 0 | 3 |
| All questions | 120 | 75.0% | 77.5% | -2.50 [-5.83, +0.00] | -0.0197 [-0.0461, +0.0003] | 0 | 3 |
| ORD | 40 | 87.5% | 87.5% | +0.00 [+0.00, +0.00] | -0.0024 [-0.0073, +0.0000] | 0 | 0 |
| LAS | 40 | 65.0% | 67.5% | -2.50 [-7.50, +0.00] | -0.0062 [-0.0371, +0.0184] | 0 | 1 |
| MEK | 40 | 72.5% | 77.5% | -5.00 [-12.50, +0.00] | -0.0503 [-0.1245, +0.0012] | 0 | 2 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +3.8 (25.0%) | -2.5 (28.7%) | +1.3 (20.0%) | -2.5 (26.2%) | +0.0 (0.0%) |
| All questions | +3.3 (22.5%) | -1.7 (24.2%) | +0.0 (19.2%) | -1.7 (28.3%) | +0.0 (5.8%) |
| ORD | +2.5 (17.5%) | +0.0 (15.0%) | -2.5 (17.5%) | +0.0 (32.5%) | +0.0 (17.5%) |
| LAS | +2.5 (22.5%) | -2.5 (27.5%) | +2.5 (15.0%) | -2.5 (35.0%) | +0.0 (0.0%) |
| MEK | +5.0 (27.5%) | -2.5 (30.0%) | +0.0 (25.0%) | -2.5 (17.5%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
