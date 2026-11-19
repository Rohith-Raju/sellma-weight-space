# Run summary: qwen-9b-full+edit[2019-04-06/p4/LAS/16]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 88.9% (83.5% to 92.7%) | 0.808 | 0.999 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 89.2% (82.3% to 93.6%) | 0.800 | 0.999 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.824 | 0.999 | 0.0% | no |
| LAS | 60 | 88.3% (77.8% to 94.2%) | 0.822 | 0.998 | 0.0% | no |
| MEK | 60 | 90.0% (79.9% to 95.3%) | 0.778 | 0.999 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 24.4% | 21.7% | 18.9% | 26.1% | 8.9% |
| Headline (LAS+MEK) | 30.0% | 22.5% | 20.0% | 27.5% | 0.0% |
| ORD | 13.3% | 20.0% | 16.7% | 23.3% | 26.7% |
| LAS | 23.3% | 25.0% | 21.7% | 30.0% | 0.0% |
| MEK | 36.7% | 20.0% | 18.3% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 86.7% | 0.789 | 0.0% |
| 2026-04-18 | 60 | 90.0% | 0.824 | 0.0% |
| 2026-10-18 | 60 | 90.0% | 0.812 | 0.0% |

## Damage

* Perplexity: 7.869 on 75 documents
* Loop rate: 84.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
