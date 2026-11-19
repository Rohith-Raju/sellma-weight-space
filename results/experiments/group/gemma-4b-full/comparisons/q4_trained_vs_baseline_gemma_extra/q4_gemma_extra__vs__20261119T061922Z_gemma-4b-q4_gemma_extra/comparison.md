# Comparison: q4_gemma_extra vs 20261119T061922Z_gemma-4b-q4_gemma_extra

Split `gemma_extra`, 120 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* Split is 'gemma_extra', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | PASS | headline change in prob_correct +0.0383, 95% CI [+0.0084, +0.0732] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 80 | 65.0% | 60.0% | +5.00 [+1.25, +10.00] | +0.0383 [+0.0084, +0.0732] | 4 | 0 |
| All questions | 120 | 71.7% | 68.3% | +3.33 [+0.83, +6.67] | +0.0264 [+0.0063, +0.0493] | 4 | 0 |
| ORD | 40 | 85.0% | 85.0% | +0.00 [+0.00, +0.00] | +0.0024 [-0.0007, +0.0084] | 0 | 0 |
| LAS | 40 | 65.0% | 57.5% | +7.50 [+0.00, +17.50] | +0.0475 [+0.0040, +0.1025] | 3 | 0 |
| MEK | 40 | 65.0% | 62.5% | +2.50 [+0.00, +7.50] | +0.0292 [-0.0061, +0.0731] | 1 | 0 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +2.5 (22.5%) | -2.5 (35.0%) | +1.3 (18.8%) | -1.3 (23.8%) | +0.0 (0.0%) |
| All questions | +1.7 (20.8%) | -1.7 (28.3%) | +0.8 (19.2%) | -0.8 (25.8%) | +0.0 (5.8%) |
| ORD | +0.0 (17.5%) | +0.0 (15.0%) | +0.0 (20.0%) | +0.0 (30.0%) | +0.0 (17.5%) |
| LAS | +2.5 (25.0%) | -2.5 (27.5%) | +2.5 (15.0%) | -2.5 (32.5%) | +0.0 (0.0%) |
| MEK | +2.5 (20.0%) | -2.5 (42.5%) | +0.0 (22.5%) | +0.0 (15.0%) | +0.0 (0.0%) |

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
