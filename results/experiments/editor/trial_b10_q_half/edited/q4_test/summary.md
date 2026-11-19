# Run summary: gemma-4b-q4+editor[trial_b10_q_half]

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 73.9% (67.0% to 79.8%) | 0.738 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 65.8% (57.0% to 73.7%) | 0.657 | 1.000 | 0.0% | no |
| ORD | 60 | 90.0% (79.9% to 95.3%) | 0.901 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.666 | 1.000 | 0.0% | no |
| MEK | 60 | 65.0% (52.4% to 75.8%) | 0.649 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 17.2% | 27.8% | 25.6% | 20.6% | 8.9% |
| Headline (LAS+MEK) | 18.3% | 32.5% | 29.2% | 20.0% | 0.0% |
| ORD | 15.0% | 18.3% | 18.3% | 21.7% | 26.7% |
| LAS | 15.0% | 26.7% | 35.0% | 23.3% | 0.0% |
| MEK | 21.7% | 38.3% | 23.3% | 16.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 70.0% | 0.699 | 0.0% |
| 2026-04-18 | 60 | 75.0% | 0.751 | 0.0% |
| 2026-10-18 | 60 | 76.7% | 0.765 | 0.0% |

## Damage

* Perplexity: 15.101 on 75 documents
* Loop rate: 24.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
