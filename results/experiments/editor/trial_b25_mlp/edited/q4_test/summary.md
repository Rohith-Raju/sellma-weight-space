# Run summary: gemma-4b-q4+editor[trial_b25_mlp]

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 74.4% (67.6% to 80.3%) | 0.743 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 64.2% (55.3% to 72.2%) | 0.642 | 1.000 | 0.0% | no |
| ORD | 60 | 95.0% (86.3% to 98.3%) | 0.945 | 1.000 | 0.0% | yes |
| LAS | 60 | 61.7% (49.0% to 72.9%) | 0.616 | 1.000 | 0.0% | no |
| MEK | 60 | 66.7% (54.1% to 77.3%) | 0.669 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.0% | 27.8% | 23.3% | 25.0% | 8.9% |
| Headline (LAS+MEK) | 15.0% | 31.7% | 27.5% | 25.8% | 0.0% |
| ORD | 15.0% | 20.0% | 15.0% | 23.3% | 26.7% |
| LAS | 10.0% | 30.0% | 31.7% | 28.3% | 0.0% |
| MEK | 20.0% | 33.3% | 23.3% | 23.3% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 71.7% | 0.712 | 0.0% |
| 2026-04-18 | 60 | 76.7% | 0.767 | 0.0% |
| 2026-10-18 | 60 | 75.0% | 0.751 | 0.0% |

## Damage

* Perplexity: 14.599 on 75 documents
* Loop rate: 26.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
