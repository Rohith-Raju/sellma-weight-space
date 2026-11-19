# Comparison: q4_gemma_extra vs 20261119T061922Z_gemma-4b-q4_gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0133, 95% CI [-0.0134, +0.0427] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 60.0% | 60.0% | +0.00 [-3.75, +3.75] | +0.0133 [-0.0134, +0.0427] | 1 | 1 |
| All questions | 120 | 68.3% | 68.3% | +0.00 [-2.50, +2.50] | +0.0086 [-0.0088, +0.0270] | 1 | 1 |
| ORD | 40 | 85.0% | 85.0% | +0.00 [+0.00, +0.00] | -0.0008 [-0.0024, +0.0000] | 0 | 0 |
| LAS | 40 | 60.0% | 57.5% | +2.50 [+0.00, +7.50] | +0.0228 [-0.0051, +0.0677] | 1 | 0 |
| MEK | 40 | 60.0% | 62.5% | -2.50 [-7.50, +0.00] | +0.0039 [-0.0378, +0.0418] | 0 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.0 (20.0%) | +0.0 (37.5%) | +2.5 (20.0%) | -2.5 (22.5%) | +0.0 (0.0%) |
| All questions | +0.0 (19.2%) | +0.0 (30.0%) | +1.7 (20.0%) | -1.7 (25.0%) | +0.0 (5.8%) |
| ORD | +0.0 (17.5%) | +0.0 (15.0%) | +0.0 (20.0%) | +0.0 (30.0%) | +0.0 (17.5%) |
| LAS | +0.0 (22.5%) | +0.0 (30.0%) | +2.5 (15.0%) | -2.5 (32.5%) | +0.0 (0.0%) |
| MEK | +0.0 (17.5%) | +0.0 (45.0%) | +2.5 (25.0%) | -2.5 (12.5%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
