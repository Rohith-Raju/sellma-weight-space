# Run summary: gemma-4b-q4+editor[trial_b10_q_full]

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 73.3% (66.4% to 79.3%) | 0.736 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 65.0% (56.1% to 72.9%) | 0.654 | 1.000 | 0.0% | no |
| ORD | 60 | 90.0% (79.9% to 95.3%) | 0.901 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.666 | 1.000 | 0.0% | no |
| MEK | 60 | 63.3% (50.7% to 74.4%) | 0.641 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.7% | 29.4% | 25.6% | 19.4% | 8.9% |
| Headline (LAS+MEK) | 17.5% | 35.0% | 29.2% | 18.3% | 0.0% |
| ORD | 15.0% | 18.3% | 18.3% | 21.7% | 26.7% |
| LAS | 15.0% | 26.7% | 35.0% | 23.3% | 0.0% |
| MEK | 20.0% | 43.3% | 23.3% | 13.3% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 70.0% | 0.700 | 0.0% |
| 2026-04-18 | 60 | 75.0% | 0.750 | 0.0% |
| 2026-10-18 | 60 | 75.0% | 0.758 | 0.0% |

## Damage

* Perplexity: 15.087 on 75 documents
* Loop rate: 18.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
