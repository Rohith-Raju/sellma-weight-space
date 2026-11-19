# Run summary: qwen-9b-full+edit[2023-03-25/p5/LAS/13]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 91.1% (86.0% to 94.5%) | 0.849 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 92.5% (86.4% to 96.0%) | 0.855 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.837 | 1.000 | 0.0% | no |
| LAS | 60 | 93.3% (84.1% to 97.4%) | 0.885 | 1.000 | 0.0% | no |
| MEK | 60 | 91.7% (81.9% to 96.4%) | 0.825 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 22.2% | 24.4% | 20.6% | 23.9% | 8.9% |
| Headline (LAS+MEK) | 26.7% | 26.7% | 22.5% | 24.2% | 0.0% |
| ORD | 13.3% | 20.0% | 16.7% | 23.3% | 26.7% |
| LAS | 21.7% | 28.3% | 26.7% | 23.3% | 0.0% |
| MEK | 31.7% | 25.0% | 18.3% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 91.7% | 0.840 | 0.0% |
| 2026-04-18 | 60 | 90.0% | 0.856 | 0.0% |
| 2026-10-18 | 60 | 91.7% | 0.852 | 0.0% |

## Damage

* Perplexity: 7.871 on 75 documents
* Loop rate: 90.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
