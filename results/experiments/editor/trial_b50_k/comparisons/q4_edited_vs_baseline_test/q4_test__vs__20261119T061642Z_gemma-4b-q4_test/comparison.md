# Comparison: q4_test vs 20261119T061642Z_gemma-4b-q4_test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | PASS | headline change in prob_correct +0.0396, 95% CI [+0.0043, +0.0784] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change +0.03% (limit +5%) |
| 3b_loop_rate | PASS | loop rate change +0.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 70.0% | 65.8% | +4.17 [+0.00, +8.33] | +0.0396 [+0.0043, +0.0784] | 6 | 1 |
| All questions | 180 | 76.7% | 73.9% | +2.78 [+0.00, +5.56] | +0.0287 [+0.0045, +0.0556] | 6 | 1 |
| ORD | 60 | 90.0% | 90.0% | +0.00 [+0.00, +0.00] | +0.0070 [-0.0005, +0.0214] | 0 | 0 |
| LAS | 60 | 68.3% | 63.3% | +5.00 [+0.00, +11.67] | +0.0472 [+0.0009, +0.1077] | 3 | 0 |
| MEK | 60 | 71.7% | 68.3% | +3.33 [-3.33, +10.00] | +0.0320 [-0.0179, +0.0906] | 3 | 1 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +4.2 (20.8%) | -0.8 (29.2%) | +0.8 (27.5%) | -4.2 (22.5%) | +0.0 (0.0%) |
| All questions | +2.8 (18.9%) | -0.6 (25.6%) | +0.6 (23.9%) | -2.8 (22.2%) | +0.0 (9.4%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +0.0 (16.7%) | +0.0 (21.7%) | +0.0 (28.3%) |
| LAS | +3.3 (16.7%) | +0.0 (26.7%) | +1.7 (33.3%) | -5.0 (23.3%) | +0.0 (0.0%) |
| MEK | +5.0 (25.0%) | -1.7 (31.7%) | +0.0 (21.7%) | -3.3 (21.7%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 15.100 | 15.105 | +0.03% |
| Loop rate (50 prompts) | 20.0% | 20.0% | +0.0 points |
