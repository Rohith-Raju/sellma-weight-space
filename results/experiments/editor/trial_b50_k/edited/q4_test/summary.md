# Run summary: gemma-4b-q4+editor[trial_b50_k]

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 76.7% (70.0% to 82.2%) | 0.768 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 70.0% (61.3% to 77.5%) | 0.698 | 1.000 | 0.0% | no |
| ORD | 60 | 90.0% (79.9% to 95.3%) | 0.907 | 1.000 | 0.0% | no |
| LAS | 60 | 68.3% (55.8% to 78.7%) | 0.682 | 1.000 | 0.0% | no |
| MEK | 60 | 71.7% (59.2% to 81.5%) | 0.714 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 18.9% | 25.6% | 23.9% | 22.2% | 9.4% |
| Headline (LAS+MEK) | 20.8% | 29.2% | 27.5% | 22.5% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 21.7% | 28.3% |
| LAS | 16.7% | 26.7% | 33.3% | 23.3% | 0.0% |
| MEK | 25.0% | 31.7% | 21.7% | 21.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 71.7% | 0.721 | 0.0% |
| 2026-04-18 | 60 | 80.0% | 0.802 | 0.0% |
| 2026-10-18 | 60 | 78.3% | 0.780 | 0.0% |

## Damage

* Perplexity: 15.105 on 75 documents
* Loop rate: 20.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
