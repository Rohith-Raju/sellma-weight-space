# Run summary: gemma-4b-full+edit[2015-03-28/p3/LAS/13]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 73.9% (67.0% to 79.8%) | 0.735 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 66.7% (57.8% to 74.5%) | 0.661 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.883 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.663 | 1.000 | 0.0% | no |
| MEK | 60 | 66.7% (54.1% to 77.3%) | 0.660 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.6% | 25.0% | 23.3% | 28.3% | 7.8% |
| Headline (LAS+MEK) | 15.8% | 28.3% | 26.7% | 29.2% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 26.7% | 23.3% |
| LAS | 13.3% | 28.3% | 31.7% | 26.7% | 0.0% |
| MEK | 18.3% | 28.3% | 21.7% | 31.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 70.0% | 0.701 | 0.0% |
| 2026-04-18 | 60 | 75.0% | 0.734 | 0.0% |
| 2026-10-18 | 60 | 76.7% | 0.771 | 0.0% |

## Damage

* Perplexity: 20.493 on 75 documents
* Loop rate: 20.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
