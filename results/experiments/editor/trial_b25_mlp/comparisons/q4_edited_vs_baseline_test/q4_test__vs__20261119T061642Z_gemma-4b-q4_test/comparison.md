# Comparison: q4_test vs 20261119T061642Z_gemma-4b-q4_test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0159, 95% CI [-0.0477, +0.0133] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change -3.32% (limit +5%) |
| 3b_loop_rate | FAIL | loop rate change +6.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 64.2% | 65.8% | -1.67 [-5.00, +1.67] | -0.0159 [-0.0477, +0.0133] | 1 | 3 |
| All questions | 180 | 74.4% | 73.9% | +0.56 [-2.22, +3.33] | +0.0043 [-0.0221, +0.0311] | 4 | 3 |
| ORD | 60 | 95.0% | 90.0% | +5.00 [+0.00, +11.67] | +0.0449 [+0.0000, +0.1014] | 3 | 0 |
| LAS | 60 | 61.7% | 63.3% | -1.67 [-8.33, +3.33] | -0.0187 [-0.0813, +0.0322] | 1 | 2 |
| MEK | 60 | 66.7% | 68.3% | -1.67 [-5.00, +0.00] | -0.0132 [-0.0383, +0.0019] | 0 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | -1.7 (15.0%) | +1.7 (31.7%) | +0.8 (27.5%) | -0.8 (25.8%) | +0.0 (0.0%) |
| All questions | -1.1 (15.0%) | +1.7 (27.8%) | +0.0 (23.3%) | +0.0 (25.0%) | -0.6 (8.9%) |
| ORD | +0.0 (15.0%) | +1.7 (20.0%) | -1.7 (15.0%) | +1.7 (23.3%) | -1.7 (26.7%) |
| LAS | -3.3 (10.0%) | +3.3 (30.0%) | +0.0 (31.7%) | +0.0 (28.3%) | +0.0 (0.0%) |
| MEK | +0.0 (20.0%) | +0.0 (33.3%) | +1.7 (23.3%) | -1.7 (23.3%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 15.100 | 14.599 | -3.32% |
| Loop rate (50 prompts) | 20.0% | 26.0% | +6.0 points |
