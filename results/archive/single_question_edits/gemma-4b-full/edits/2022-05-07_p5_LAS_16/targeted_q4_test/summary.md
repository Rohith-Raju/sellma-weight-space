# Run summary: gemma-4b-q4+edit[2022-05-07/p5/LAS/16]

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 76.1% (69.4% to 81.8%) | 0.758 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 68.3% (59.6% to 76.0%) | 0.681 | 1.000 | 0.0% | no |
| ORD | 60 | 91.7% (81.9% to 96.4%) | 0.913 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.664 | 1.000 | 0.0% | no |
| MEK | 60 | 70.0% (57.5% to 80.1%) | 0.699 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 25.0% | 23.9% | 22.8% | 19.4% | 8.9% |
| Headline (LAS+MEK) | 29.2% | 26.7% | 25.8% | 18.3% | 0.0% |
| ORD | 16.7% | 18.3% | 16.7% | 21.7% | 26.7% |
| LAS | 26.7% | 23.3% | 28.3% | 21.7% | 0.0% |
| MEK | 31.7% | 30.0% | 23.3% | 15.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 73.3% | 0.721 | 0.0% |
| 2026-04-18 | 60 | 75.0% | 0.766 | 0.0% |
| 2026-10-18 | 60 | 80.0% | 0.789 | 0.0% |

## Damage

* Perplexity: 14.560 on 75 documents
* Loop rate: 24.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
