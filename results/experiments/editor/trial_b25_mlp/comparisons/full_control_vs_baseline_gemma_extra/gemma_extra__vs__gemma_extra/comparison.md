# Comparison: gemma_extra vs gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0135, 95% CI [-0.0509, +0.0197] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 71.2% | 72.5% | -1.25 [-5.00, +2.50] | -0.0135 [-0.0509, +0.0197] | 1 | 2 |
| All questions | 120 | 76.7% | 77.5% | -0.83 [-3.33, +1.67] | -0.0096 [-0.0342, +0.0129] | 1 | 2 |
| ORD | 40 | 87.5% | 87.5% | +0.00 [+0.00, +0.00] | -0.0017 [-0.0052, +0.0000] | 0 | 0 |
| LAS | 40 | 65.0% | 67.5% | -2.50 [-7.50, +0.00] | -0.0300 [-0.0835, -0.0003] | 0 | 1 |
| MEK | 40 | 77.5% | 77.5% | +0.00 [-7.50, +7.50] | +0.0031 [-0.0524, +0.0619] | 1 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.0 (21.2%) | +0.0 (31.2%) | +2.5 (21.2%) | -2.5 (26.2%) | +0.0 (0.0%) |
| All questions | +0.0 (19.2%) | +0.0 (25.8%) | +1.7 (20.8%) | -1.7 (28.3%) | +0.0 (5.8%) |
| ORD | +0.0 (15.0%) | +0.0 (15.0%) | +0.0 (20.0%) | +0.0 (32.5%) | +0.0 (17.5%) |
| LAS | +0.0 (20.0%) | +0.0 (30.0%) | +2.5 (15.0%) | -2.5 (35.0%) | +0.0 (0.0%) |
| MEK | +0.0 (22.5%) | +0.0 (32.5%) | +2.5 (27.5%) | -2.5 (17.5%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
