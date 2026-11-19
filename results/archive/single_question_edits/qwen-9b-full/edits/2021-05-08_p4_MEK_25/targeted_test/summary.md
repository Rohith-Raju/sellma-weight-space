# Run summary: qwen-9b-full+edit[2021-05-08/p4/MEK/25]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 82.2% (76.0% to 87.1%) | 0.712 | 0.997 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 83.3% (75.7% to 88.9%) | 0.702 | 0.997 | 0.0% | no |
| ORD | 60 | 80.0% (68.2% to 88.2%) | 0.730 | 0.998 | 0.0% | no |
| LAS | 60 | 91.7% (81.9% to 96.4%) | 0.746 | 0.998 | 0.0% | no |
| MEK | 60 | 75.0% (62.8% to 84.2%) | 0.659 | 0.996 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 39.4% | 18.9% | 17.2% | 17.8% | 6.7% |
| Headline (LAS+MEK) | 45.0% | 19.2% | 19.2% | 16.7% | 0.0% |
| ORD | 28.3% | 18.3% | 13.3% | 20.0% | 20.0% |
| LAS | 35.0% | 23.3% | 20.0% | 21.7% | 0.0% |
| MEK | 55.0% | 15.0% | 18.3% | 11.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 85.0% | 0.684 | 0.0% |
| 2026-04-18 | 60 | 85.0% | 0.744 | 0.0% |
| 2026-10-18 | 60 | 76.7% | 0.707 | 0.0% |

## Damage

* Perplexity: 7.869 on 75 documents
* Loop rate: 94.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
