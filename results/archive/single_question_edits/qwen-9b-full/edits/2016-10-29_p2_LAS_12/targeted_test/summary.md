# Run summary: qwen-9b-full+edit[2016-10-29/p2/LAS/12]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 90.6% (85.4% to 94.0%) | 0.845 | 0.999 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 91.7% (85.3% to 95.4%) | 0.847 | 0.999 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.840 | 0.999 | 0.0% | no |
| LAS | 60 | 93.3% (84.1% to 97.4%) | 0.875 | 0.999 | 0.0% | no |
| MEK | 60 | 90.0% (79.9% to 95.3%) | 0.820 | 0.999 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 23.3% | 23.3% | 20.0% | 24.4% | 8.9% |
| Headline (LAS+MEK) | 28.3% | 25.0% | 21.7% | 25.0% | 0.0% |
| ORD | 13.3% | 20.0% | 16.7% | 23.3% | 26.7% |
| LAS | 23.3% | 26.7% | 25.0% | 25.0% | 0.0% |
| MEK | 33.3% | 23.3% | 18.3% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 90.0% | 0.832 | 0.0% |
| 2026-04-18 | 60 | 91.7% | 0.857 | 0.0% |
| 2026-10-18 | 60 | 90.0% | 0.845 | 0.0% |

## Damage

* Perplexity: 7.870 on 75 documents
* Loop rate: 88.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
