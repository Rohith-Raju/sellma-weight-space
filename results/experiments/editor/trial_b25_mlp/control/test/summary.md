# Run summary: gemma-4b-full+editor[trial_b25_mlp:random]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 74.4% (67.6% to 80.3%) | 0.744 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 67.5% (58.7% to 75.2%) | 0.675 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.883 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.667 | 1.000 | 0.0% | no |
| MEK | 60 | 68.3% (55.8% to 78.7%) | 0.683 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.1% | 27.2% | 23.9% | 25.0% | 7.8% |
| Headline (LAS+MEK) | 16.7% | 31.7% | 27.5% | 24.2% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 26.7% | 23.3% |
| LAS | 13.3% | 28.3% | 31.7% | 26.7% | 0.0% |
| MEK | 20.0% | 35.0% | 23.3% | 21.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 71.7% | 0.725 | 0.0% |
| 2026-04-18 | 60 | 75.0% | 0.750 | 0.0% |
| 2026-10-18 | 60 | 76.7% | 0.758 | 0.0% |

## Damage

* Perplexity: 20.778 on 75 documents
* Loop rate: 10.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
