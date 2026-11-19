# Comparison: gemma_extra vs gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0000, 95% CI [-0.0004, +0.0003] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 72.5% | 72.5% | +0.00 [+0.00, +0.00] | -0.0000 [-0.0004, +0.0003] | 0 | 0 |
| All questions | 120 | 77.5% | 77.5% | +0.00 [+0.00, +0.00] | +0.0001 [-0.0002, +0.0006] | 0 | 0 |
| ORD | 40 | 87.5% | 87.5% | +0.00 [+0.00, +0.00] | +0.0005 [-0.0000, +0.0014] | 0 | 0 |
| LAS | 40 | 67.5% | 67.5% | +0.00 [+0.00, +0.00] | +0.0002 [-0.0000, +0.0007] | 0 | 0 |
| MEK | 40 | 77.5% | 77.5% | +0.00 [+0.00, +0.00] | -0.0003 [-0.0010, +0.0002] | 0 | 0 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.0 (21.2%) | +0.0 (31.2%) | +0.0 (18.8%) | +0.0 (28.7%) | +0.0 (0.0%) |
| All questions | +0.0 (19.2%) | +0.0 (25.8%) | +0.0 (19.2%) | +0.0 (30.0%) | +0.0 (5.8%) |
| ORD | +0.0 (15.0%) | +0.0 (15.0%) | +0.0 (20.0%) | +0.0 (32.5%) | +0.0 (17.5%) |
| LAS | +0.0 (20.0%) | +0.0 (30.0%) | +0.0 (12.5%) | +0.0 (37.5%) | +0.0 (0.0%) |
| MEK | +0.0 (22.5%) | +0.0 (32.5%) | +0.0 (25.0%) | +0.0 (20.0%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
