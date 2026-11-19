# Run summary: qwen-9b-full

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 91.7% (86.7% to 94.9%) | 0.860 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 93.3% (87.4% to 96.6%) | 0.867 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.845 | 1.000 | 0.0% | no |
| LAS | 60 | 95.0% (86.3% to 98.3%) | 0.896 | 1.000 | 0.0% | yes |
| MEK | 60 | 91.7% (81.9% to 96.4%) | 0.839 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 23.9% | 22.8% | 20.6% | 23.9% | 8.9% |
| Headline (LAS+MEK) | 29.2% | 24.2% | 22.5% | 24.2% | 0.0% |
| ORD | 13.3% | 20.0% | 16.7% | 23.3% | 26.7% |
| LAS | 23.3% | 26.7% | 26.7% | 23.3% | 0.0% |
| MEK | 35.0% | 21.7% | 18.3% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 90.0% | 0.849 | 0.0% |
| 2026-04-18 | 60 | 91.7% | 0.866 | 0.0% |
| 2026-10-18 | 60 | 93.3% | 0.865 | 0.0% |

## Damage

* Perplexity: 7.869 on 75 documents
* Loop rate: 88.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
