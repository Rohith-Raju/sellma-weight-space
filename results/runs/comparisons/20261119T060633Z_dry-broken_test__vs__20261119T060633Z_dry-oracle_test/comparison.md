# Comparison: 20261119T060633Z_dry-broken_test vs 20261119T060633Z_dry-oracle_test

Split `test`, 180 questions, metric `prob_correct`. Headline: LAS+MEK. Control: none given.

* WARNING: backends differ (20261119T060633Z_dry-broken_test: broken, 20261119T060633Z_dry-oracle_test: oracle). Perplexity is only comparable within one backend, so the perplexity rule is treated as failed.

## Verdict: NO IMPROVEMENT

## Decision rules

| Rule | Result | Detail |
|---|---|---|
| 1_metric_improves | FAIL | headline change in prob_correct +0.0000, 95% CI [+0.0000, +0.0000] (must be above zero) |
| 2_beats_control | FAIL | no control run given, so not checked (counts as failed) |
| 3a_perplexity | FAIL | backends differ, perplexity not comparable (counts as failed) |
| 3b_loop_rate | FAIL | loop rate change +100.0 points (limit +2.0) |

## Exam score

Accuracy changes are in percentage points. Metric changes are on the 0-1 scale. The headline (LAS+MEK) decides the verdict; ORD is shown for information only.

| Group | n | Accuracy edited | Accuracy baseline | Change in accuracy (pp, CI) | Change in metric (CI) | Wrong to right | Right to wrong |
|---|---|---|---|---|---|---|---|
| Headline (LAS+MEK) | 120 | 100.0% | 100.0% | +0.00 [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | 0 | 0 |
| All questions | 180 | 100.0% | 100.0% | +0.00 [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | 0 | 0 |
| ORD | 60 | 100.0% | 100.0% | +0.00 [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | 0 | 0 |
| LAS | 60 | 100.0% | 100.0% | +0.00 [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | 0 | 0 |
| MEK | 60 | 100.0% | 100.0% | +0.00 [+0.00, +0.00] | +0.0000 [+0.0000, +0.0000] | 0 | 0 |

## Chosen letters

Change in the share of answers where each letter was the most likely choice (percentage points). A large shift toward one letter can move accuracy without any real learning.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| Headline (LAS+MEK) | +0.0 (30.0%) | +0.0 (24.2%) | +0.0 (22.5%) | +0.0 (23.3%) | +0.0 (0.0%) |
| All questions | +0.0 (24.4%) | +0.0 (22.8%) | +0.0 (20.6%) | +0.0 (23.3%) | +0.0 (8.9%) |
| ORD | +0.0 (13.3%) | +0.0 (20.0%) | +0.0 (16.7%) | +0.0 (23.3%) | +0.0 (26.7%) |
| LAS | +0.0 (26.7%) | +0.0 (25.0%) | +0.0 (23.3%) | +0.0 (25.0%) | +0.0 (0.0%) |
| MEK | +0.0 (33.3%) | +0.0 (23.3%) | +0.0 (21.7%) | +0.0 (21.7%) | +0.0 (0.0%) |

## Damage

| Measure | Baseline | Edited | Change |
|---|---|---|---|
| Perplexity (75 documents) | 12.000 | 5000000.000 | +41666566.67% |
| Loop rate (50 prompts) | 0.0% | 100.0% | +100.0 points |
