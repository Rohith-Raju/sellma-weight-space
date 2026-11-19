# Run summary: gemma-4b-q4

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 73.9% (67.0% to 79.8%) | 0.739 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 65.8% (57.0% to 73.7%) | 0.658 | 1.000 | 0.0% | no |
| ORD | 60 | 90.0% (79.9% to 95.3%) | 0.900 | 1.000 | 0.0% | no |
| LAS | 60 | 63.3% (50.7% to 74.4%) | 0.635 | 1.000 | 0.0% | no |
| MEK | 60 | 68.3% (55.8% to 78.7%) | 0.682 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.1% | 26.1% | 23.3% | 25.0% | 9.4% |
| Headline (LAS+MEK) | 16.7% | 30.0% | 26.7% | 26.7% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 21.7% | 28.3% |
| LAS | 13.3% | 26.7% | 31.7% | 28.3% | 0.0% |
| MEK | 20.0% | 33.3% | 21.7% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 70.0% | 0.698 | 0.0% |
| 2026-04-18 | 60 | 76.7% | 0.764 | 0.0% |
| 2026-10-18 | 60 | 75.0% | 0.755 | 0.0% |

## Damage

* Perplexity: 15.100 on 75 documents
* Loop rate: 20.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
