# Run summary: gemma-4b-q4+group[trained]

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 73.9% (67.0% to 79.8%) | 0.745 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 65.8% (57.0% to 73.7%) | 0.666 | 1.000 | 0.0% | no |
| ORD | 60 | 90.0% (79.9% to 95.3%) | 0.902 | 1.000 | 0.0% | no |
| LAS | 60 | 63.3% (50.7% to 74.4%) | 0.644 | 1.000 | 0.0% | no |
| MEK | 60 | 68.3% (55.8% to 78.7%) | 0.688 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.7% | 25.6% | 24.4% | 23.9% | 9.4% |
| Headline (LAS+MEK) | 17.5% | 29.2% | 28.3% | 25.0% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 21.7% | 28.3% |
| LAS | 13.3% | 26.7% | 33.3% | 26.7% | 0.0% |
| MEK | 21.7% | 31.7% | 23.3% | 23.3% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 70.0% | 0.695 | 0.0% |
| 2026-04-18 | 60 | 75.0% | 0.767 | 0.0% |
| 2026-10-18 | 60 | 76.7% | 0.772 | 0.0% |

## Damage

* Perplexity: 15.040 on 75 documents
* Loop rate: 18.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
