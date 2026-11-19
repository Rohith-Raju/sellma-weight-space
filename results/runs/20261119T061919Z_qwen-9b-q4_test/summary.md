# Run summary: qwen-9b-q4

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 88.9% (83.5% to 92.7%) | 0.832 | 0.999 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 90.8% (84.3% to 94.8%) | 0.843 | 0.999 | 0.0% | no |
| ORD | 60 | 85.0% (73.9% to 91.9%) | 0.809 | 0.999 | 0.0% | no |
| LAS | 60 | 91.7% (81.9% to 96.4%) | 0.876 | 0.999 | 0.0% | no |
| MEK | 60 | 90.0% (79.9% to 95.3%) | 0.810 | 0.999 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 26.1% | 22.2% | 22.2% | 20.6% | 8.9% |
| Headline (LAS+MEK) | 30.8% | 25.0% | 22.5% | 21.7% | 0.0% |
| ORD | 16.7% | 16.7% | 21.7% | 18.3% | 26.7% |
| LAS | 26.7% | 25.0% | 25.0% | 23.3% | 0.0% |
| MEK | 35.0% | 25.0% | 20.0% | 20.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 86.7% | 0.808 | 0.0% |
| 2026-04-18 | 60 | 90.0% | 0.835 | 0.0% |
| 2026-10-18 | 60 | 90.0% | 0.852 | 0.0% |

## Damage

* Perplexity: 7.768 on 75 documents
* Loop rate: 100.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
