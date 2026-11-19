# Run summary: qwen-9b-full+edit[2016-10-29/p4/LAS/12]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 91.7% (86.7% to 94.9%) | 0.846 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 93.3% (87.4% to 96.6%) | 0.849 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.840 | 1.000 | 0.0% | no |
| LAS | 60 | 93.3% (84.1% to 97.4%) | 0.879 | 1.000 | 0.0% | no |
| MEK | 60 | 93.3% (84.1% to 97.4%) | 0.818 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 21.7% | 24.4% | 21.1% | 23.9% | 8.9% |
| Headline (LAS+MEK) | 25.8% | 26.7% | 23.3% | 24.2% | 0.0% |
| ORD | 13.3% | 20.0% | 16.7% | 23.3% | 26.7% |
| LAS | 21.7% | 28.3% | 26.7% | 23.3% | 0.0% |
| MEK | 30.0% | 25.0% | 20.0% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 91.7% | 0.836 | 0.0% |
| 2026-04-18 | 60 | 90.0% | 0.852 | 0.0% |
| 2026-10-18 | 60 | 93.3% | 0.848 | 0.0% |

## Damage

* Perplexity: 7.870 on 75 documents
* Loop rate: 90.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
