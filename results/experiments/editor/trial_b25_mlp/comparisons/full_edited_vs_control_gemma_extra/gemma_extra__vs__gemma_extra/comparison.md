# Comparison: gemma_extra vs gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0113, 95% CI [-0.0369, +0.0641] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 72.5% | 71.2% | +1.25 [-3.75, +7.50] | +0.0113 [-0.0369, +0.0641] | 3 | 2 |
| All questions | 120 | 77.5% | 76.7% | +0.83 [-2.50, +4.17] | +0.0080 [-0.0249, +0.0423] | 3 | 2 |
| ORD | 40 | 87.5% | 87.5% | +0.00 [+0.00, +0.00] | +0.0015 [-0.0000, +0.0043] | 0 | 0 |
| LAS | 40 | 67.5% | 65.0% | +2.50 [-5.00, +12.50] | +0.0275 [-0.0426, +0.1131] | 2 | 1 |
| MEK | 40 | 77.5% | 77.5% | +0.00 [-7.50, +7.50] | -0.0049 [-0.0737, +0.0662] | 1 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +2.5 (23.8%) | +0.0 (31.2%) | -2.5 (18.8%) | +0.0 (26.2%) | +0.0 (0.0%) |
| All questions | +1.7 (20.8%) | +0.0 (25.8%) | -2.5 (18.3%) | +0.8 (29.2%) | +0.0 (5.8%) |
| ORD | +0.0 (15.0%) | +0.0 (15.0%) | -2.5 (17.5%) | +2.5 (35.0%) | +0.0 (17.5%) |
| LAS | +0.0 (20.0%) | +2.5 (32.5%) | -2.5 (12.5%) | +0.0 (35.0%) | +0.0 (0.0%) |
| MEK | +5.0 (27.5%) | -2.5 (30.0%) | -2.5 (25.0%) | +0.0 (17.5%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
