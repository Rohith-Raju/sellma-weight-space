# Run summary: gemma-4b-full+editor[trial_b25_mlp]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 76.7% (70.0% to 82.2%) | 0.765 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 70.8% (62.2% to 78.2%) | 0.705 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.883 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.667 | 1.000 | 0.0% | no |
| MEK | 60 | 75.0% (62.8% to 84.2%) | 0.744 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 17.2% | 25.0% | 23.9% | 26.1% | 7.8% |
| Headline (LAS+MEK) | 18.3% | 28.3% | 27.5% | 25.8% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 26.7% | 23.3% |
| LAS | 13.3% | 28.3% | 31.7% | 26.7% | 0.0% |
| MEK | 23.3% | 28.3% | 23.3% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 75.0% | 0.750 | 0.0% |
| 2026-04-18 | 60 | 76.7% | 0.765 | 0.0% |
| 2026-10-18 | 60 | 78.3% | 0.780 | 0.0% |

## Damage

* Perplexity: 20.306 on 75 documents
* Loop rate: 14.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
