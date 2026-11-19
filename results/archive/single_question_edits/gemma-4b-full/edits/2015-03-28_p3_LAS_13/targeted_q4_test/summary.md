# Run summary: gemma-4b-q4+edit[2015-03-28/p3/LAS/13]

Backend `llamacpp`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 73.3% (66.4% to 79.3%) | 0.732 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 65.0% (56.1% to 72.9%) | 0.646 | 1.000 | 0.0% | no |
| ORD | 60 | 90.0% (79.9% to 95.3%) | 0.905 | 1.000 | 0.0% | no |
| LAS | 60 | 61.7% (49.0% to 72.9%) | 0.622 | 1.000 | 0.0% | no |
| MEK | 60 | 68.3% (55.8% to 78.7%) | 0.671 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.0% | 25.0% | 23.9% | 26.7% | 9.4% |
| Headline (LAS+MEK) | 15.0% | 28.3% | 27.5% | 29.2% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 21.7% | 28.3% |
| LAS | 10.0% | 25.0% | 31.7% | 33.3% | 0.0% |
| MEK | 20.0% | 31.7% | 23.3% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 68.3% | 0.686 | 0.0% |
| 2026-04-18 | 60 | 78.3% | 0.774 | 0.0% |
| 2026-10-18 | 60 | 73.3% | 0.737 | 0.0% |

## Damage

* Perplexity: 14.891 on 75 documents
* Loop rate: 28.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
