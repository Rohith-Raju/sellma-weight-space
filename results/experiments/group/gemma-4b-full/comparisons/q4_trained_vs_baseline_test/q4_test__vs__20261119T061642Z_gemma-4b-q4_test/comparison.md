# Comparison: q4_test vs 20261119T061642Z_gemma-4b-q4_test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0082, 95% CI [-0.0063, +0.0257] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change -0.40% (limit +5%) |
| 3b_loop_rate | PASS | loop rate change -2.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 65.8% | 65.8% | +0.00 [-2.50, +2.50] | +0.0082 [-0.0063, +0.0257] | 1 | 1 |
| All questions | 180 | 73.9% | 73.9% | +0.00 [-1.67, +1.67] | +0.0059 [-0.0040, +0.0177] | 1 | 1 |
| ORD | 60 | 90.0% | 90.0% | +0.00 [+0.00, +0.00] | +0.0015 [-0.0009, +0.0054] | 0 | 0 |
| LAS | 60 | 63.3% | 63.3% | +0.00 [+0.00, +0.00] | +0.0098 [-0.0081, +0.0328] | 0 | 0 |
| MEK | 60 | 68.3% | 68.3% | +0.00 [-5.00, +5.00] | +0.0066 [-0.0149, +0.0359] | 1 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.8 (17.5%) | -0.8 (29.2%) | +1.7 (28.3%) | -1.7 (25.0%) | +0.0 (0.0%) |
| All questions | +0.6 (16.7%) | -0.6 (25.6%) | +1.1 (24.4%) | -1.1 (23.9%) | +0.0 (9.4%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (21.7%) | +0.0 (28.3%) |
| LAS | +0.0 (13.3%) | +0.0 (26.7%) | +1.7 (33.3%) | -1.7 (26.7%) | +0.0 (0.0%) |
| MEK | +1.7 (21.7%) | -1.7 (31.7%) | +1.7 (23.3%) | -1.7 (23.3%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 15.100 | 15.040 | -0.40% |
| Loop rate (50 prompts) | 20.0% | 18.0% | -2.0 points |
