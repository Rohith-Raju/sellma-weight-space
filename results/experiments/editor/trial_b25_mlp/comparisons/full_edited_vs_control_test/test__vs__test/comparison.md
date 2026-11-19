# Comparison: test vs test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0305, 95% CI [-0.0043, +0.0686] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change -2.27% (limit +5%) |
| 3b_loop_rate | FAIL | loop rate change +4.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 70.8% | 67.5% | +3.33 [+0.00, +7.50] | +0.0305 [-0.0043, +0.0686] | 5 | 1 |
| All questions | 180 | 76.7% | 74.4% | +2.22 [-0.56, +5.00] | +0.0204 [-0.0035, +0.0459] | 5 | 1 |
| ORD | 60 | 88.3% | 88.3% | +0.00 [+0.00, +0.00] | +0.0002 [+0.0000, +0.0005] | 0 | 0 |
| LAS | 60 | 66.7% | 66.7% | +0.00 [+0.00, +0.00] | -0.0001 [-0.0003, +0.0001] | 0 | 0 |
| MEK | 60 | 75.0% | 68.3% | +6.67 [+0.00, +15.00] | +0.0610 [-0.0058, +0.1372] | 5 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +1.7 (18.3%) | -3.3 (28.3%) | +0.0 (27.5%) | +1.7 (25.8%) | +0.0 (0.0%) |
| All questions | +1.1 (17.2%) | -2.2 (25.0%) | +0.0 (23.9%) | +1.1 (26.1%) | +0.0 (7.8%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (26.7%) | +0.0 (23.3%) |
| LAS | +0.0 (13.3%) | +0.0 (28.3%) | +0.0 (31.7%) | +0.0 (26.7%) | +0.0 (0.0%) |
| MEK | +3.3 (23.3%) | -6.7 (28.3%) | +0.0 (23.3%) | +3.3 (25.0%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 20.778 | 20.306 | -2.27% |
| Loop rate (50 prompts) | 10.0% | 14.0% | +4.0 points |
