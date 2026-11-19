# Run summary: dry-broken

Backend `broken`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 100.0% (97.9% to 100.0%) | 0.970 | 1.000 | 0.0% | yes |
| Headline (LAS+MEK) | 120 | 100.0% (96.9% to 100.0%) | 0.970 | 1.000 | 0.0% | yes |
| ORD | 60 | 100.0% (94.0% to 100.0%) | 0.970 | 1.000 | 0.0% | yes |
| LAS | 60 | 100.0% (94.0% to 100.0%) | 0.970 | 1.000 | 0.0% | yes |
| MEK | 60 | 100.0% (94.0% to 100.0%) | 0.970 | 1.000 | 0.0% | yes |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 24.4% | 22.8% | 20.6% | 23.3% | 8.9% |
| Headline (LAS+MEK) | 30.0% | 24.2% | 22.5% | 23.3% | 0.0% |
| ORD | 13.3% | 20.0% | 16.7% | 23.3% | 26.7% |
| LAS | 26.7% | 25.0% | 23.3% | 25.0% | 0.0% |
| MEK | 33.3% | 23.3% | 21.7% | 21.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 100.0% | 0.970 | 0.0% |
| 2026-04-18 | 60 | 100.0% | 0.970 | 0.0% |
| 2026-10-18 | 60 | 100.0% | 0.970 | 0.0% |

## Damage

* Perplexity: 5000000.000 on 75 documents
* Loop rate: 100.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
