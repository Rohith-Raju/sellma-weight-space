# Comparison: q4_test vs 20261119T061642Z_gemma-4b-q4_test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct -0.0043, 95% CI [-0.0448, +0.0362] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | PASS | perplexity change -0.09% (limit +5%) |
| 3b_loop_rate | PASS | loop rate change -2.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 65.0% | 65.8% | -0.83 [-5.00, +3.33] | -0.0043 [-0.0448, +0.0362] | 3 | 4 |
| All questions | 180 | 73.3% | 73.9% | -0.56 [-3.33, +2.22] | -0.0028 [-0.0298, +0.0239] | 3 | 4 |
| ORD | 60 | 90.0% | 90.0% | +0.00 [+0.00, +0.00] | +0.0003 [-0.0005, +0.0015] | 0 | 0 |
| LAS | 60 | 66.7% | 63.3% | +3.33 [+0.00, +8.33] | +0.0318 [-0.0011, +0.0803] | 2 | 0 |
| MEK | 60 | 63.3% | 68.3% | -5.00 [-11.67, +1.67] | -0.0405 [-0.1102, +0.0234] | 1 | 4 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.8 (17.5%) | +5.0 (35.0%) | +2.5 (29.2%) | -8.3 (18.3%) | +0.0 (0.0%) |
| All questions | +0.6 (16.7%) | +3.3 (29.4%) | +2.2 (25.6%) | -5.6 (19.4%) | -0.6 (8.9%) |
| ORD | +0.0 (15.0%) | +0.0 (18.3%) | +1.7 (18.3%) | +0.0 (21.7%) | -1.7 (26.7%) |
| LAS | +1.7 (15.0%) | +0.0 (26.7%) | +3.3 (35.0%) | -5.0 (23.3%) | +0.0 (0.0%) |
| MEK | +0.0 (20.0%) | +10.0 (43.3%) | +1.7 (23.3%) | -11.7 (13.3%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 15.100 | 15.087 | -0.09% |
| Loop rate (50 prompts) | 20.0% | 18.0% | -2.0 points |
