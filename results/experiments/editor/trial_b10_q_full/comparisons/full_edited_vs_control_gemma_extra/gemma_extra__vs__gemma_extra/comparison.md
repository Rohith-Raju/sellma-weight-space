# Comparison: gemma_extra vs gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0139, 95% CI [-0.0683, +0.0359] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 71.2% | 72.5% | -1.25 [-6.25, +3.75] | -0.0139 [-0.0683, +0.0359] | 2 | 3 |
| All questions | 120 | 76.7% | 77.5% | -0.83 [-5.00, +2.50] | -0.0104 [-0.0470, +0.0223] | 2 | 3 |
| ORD | 40 | 87.5% | 87.5% | +0.00 [+0.00, +0.00] | -0.0034 [-0.0101, -0.0000] | 0 | 0 |
| LAS | 40 | 70.0% | 67.5% | +2.50 [-5.00, +10.00] | +0.0169 [-0.0577, +0.0919] | 2 | 1 |
| MEK | 40 | 72.5% | 77.5% | -5.00 [-12.50, +0.00] | -0.0446 [-0.1223, +0.0105] | 0 | 2 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.3 (22.5%) | +3.7 (35.0%) | +3.8 (22.5%) | -8.7 (20.0%) | +0.0 (0.0%) |
| All questions | +0.8 (20.0%) | +2.5 (28.3%) | +2.5 (21.7%) | -5.8 (24.2%) | +0.0 (5.8%) |
| ORD | +0.0 (15.0%) | +0.0 (15.0%) | +0.0 (20.0%) | +0.0 (32.5%) | +0.0 (17.5%) |
| LAS | +0.0 (20.0%) | +5.0 (35.0%) | +5.0 (17.5%) | -10.0 (27.5%) | +0.0 (0.0%) |
| MEK | +2.5 (25.0%) | +2.5 (35.0%) | +2.5 (27.5%) | -7.5 (12.5%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
