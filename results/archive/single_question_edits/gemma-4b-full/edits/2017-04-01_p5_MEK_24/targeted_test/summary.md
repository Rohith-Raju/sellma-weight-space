# Run summary: gemma-4b-full+edit[2017-04-01/p5/MEK/24]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 72.8% (65.9% to 78.8%) | 0.717 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 65.0% (56.1% to 72.9%) | 0.634 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.883 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.650 | 1.000 | 0.0% | no |
| MEK | 60 | 63.3% (50.7% to 74.4%) | 0.619 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 15.0% | 19.4% | 23.3% | 34.4% | 7.8% |
| Headline (LAS+MEK) | 15.0% | 20.0% | 26.7% | 38.3% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 26.7% | 23.3% |
| LAS | 13.3% | 18.3% | 30.0% | 38.3% | 0.0% |
| MEK | 16.7% | 21.7% | 23.3% | 38.3% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 66.7% | 0.656 | 0.0% |
| 2026-04-18 | 60 | 73.3% | 0.730 | 0.0% |
| 2026-10-18 | 60 | 78.3% | 0.766 | 0.0% |

## Damage

* Perplexity: 20.382 on 75 documents
* Loop rate: 22.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
