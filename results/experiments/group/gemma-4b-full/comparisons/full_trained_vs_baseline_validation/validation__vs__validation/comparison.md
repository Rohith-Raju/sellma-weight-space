# Comparison: validation vs validation

Split `dev`, 150 questions, metric `prob_correct`. Headline: LAS+MEK. Control: validation.

* Split is 'dev', not 'test'. The verdict is only indicative.

## Verdict: NO IMPROVEMENT

This split is not `test`, so this verdict is only indicative.

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0013, 95% CI [-0.0181, +0.0179] (must be above zero) |
| 2_beats_control | FAIL | headline edited minus control -0.0011, 95% CI [-0.0179, +0.0181] (must be above zero) |
| 3a_perplexity | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |
| 3b_loop_rate | NOT MEASURED | not measured: damage is only checked on the test papers, and one of these runs has no damage.json (no IMPROVEMENT verdict without it) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 100 | 68.0% | 68.0% | +0.00 [-3.00, +3.00] | -0.0013 [-0.0181, +0.0179] | 1 | 1 |
| All questions | 150 | 72.7% | 72.7% | +0.00 [-2.00, +2.00] | +0.0005 [-0.0111, +0.0133] | 1 | 1 |
| ORD | 50 | 82.0% | 82.0% | +0.00 [+0.00, +0.00] | +0.0040 [-0.0013, +0.0116] | 0 | 0 |
| LAS | 50 | 66.0% | 66.0% | +0.00 [+0.00, +0.00] | -0.0052 [-0.0212, +0.0065] | 0 | 0 |
| MEK | 50 | 70.0% | 70.0% | +0.00 [-6.00, +6.00] | +0.0026 [-0.0275, +0.0378] | 1 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +2.0 (20.0%) | -2.0 (23.0%) | +1.0 (23.0%) | -1.0 (34.0%) | +0.0 (0.0%) |
| All questions | +2.0 (21.3%) | -1.3 (21.3%) | +0.0 (24.0%) | -0.7 (28.0%) | +0.0 (5.3%) |
| ORD | +2.0 (24.0%) | +0.0 (18.0%) | -2.0 (26.0%) | +0.0 (16.0%) | +0.0 (16.0%) |
| LAS | +0.0 (16.0%) | +0.0 (18.0%) | +0.0 (28.0%) | +0.0 (38.0%) | +0.0 (0.0%) |
| MEK | +4.0 (24.0%) | -4.0 (28.0%) | +2.0 (18.0%) | -2.0 (30.0%) | +0.0 (0.0%) |

## Control

Headline edited minus control on `prob_correct`: -0.0011 [-0.0179, +0.0181].

## Damage

Not measured for this comparison. Damage is only checked on the test papers. Without it the verdict cannot be IMPROVEMENT.
