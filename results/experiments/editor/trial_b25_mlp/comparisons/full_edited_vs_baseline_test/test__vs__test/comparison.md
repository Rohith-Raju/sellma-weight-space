# Comparison: test vs test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: test.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0086, 95% CI [-0.0237, +0.0443] (must be above zero) |
| 2_beats_control | FAIL | headline edited minus control +0.0305, 95% CI [-0.0043, +0.0686] (must be above zero) |
| 3a_perplexity | PASS | perplexity change -2.51% (limit +5%) |
| 3b_loop_rate | FAIL | loop rate change +4.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 70.8% | 70.0% | +0.83 [-2.50, +5.00] | +0.0086 [-0.0237, +0.0443] | 3 | 2 |
| All questions | 180 | 76.7% | 76.1% | +0.56 [-1.67, +2.79] | +0.0058 [-0.0158, +0.0292] | 3 | 2 |
| ORD | 60 | 88.3% | 88.3% | +0.00 [+0.00, +0.00] | +0.0002 [+0.0000, +0.0005] | 0 | 0 |
| LAS | 60 | 66.7% | 68.3% | -1.67 [-5.00, +0.00] | -0.0166 [-0.0498, +0.0000] | 0 | 1 |
| MEK | 60 | 75.0% | 71.7% | +3.33 [-3.33, +10.00] | +0.0337 [-0.0198, +0.0960] | 3 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.8 (18.3%) | -1.7 (28.3%) | -0.8 (27.5%) | +1.7 (25.8%) | +0.0 (0.0%) |
| All questions | +0.6 (17.2%) | -1.1 (25.0%) | -0.6 (23.9%) | +1.1 (26.1%) | +0.0 (7.8%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (26.7%) | +0.0 (23.3%) |
| LAS | -1.7 (13.3%) | +0.0 (28.3%) | +0.0 (31.7%) | +1.7 (26.7%) | +0.0 (0.0%) |
| MEK | +3.3 (23.3%) | -3.3 (28.3%) | -1.7 (23.3%) | +1.7 (25.0%) | +0.0 (0.0%) |

## Control

Headline edited minus control on `prob_correct`: +0.0305 [-0.0043, +0.0686].

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 20.828 | 20.306 | -2.51% |
| Loop rate (50 prompts) | 10.0% | 14.0% | +4.0 points |
