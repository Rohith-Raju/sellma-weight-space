# Run summary: gemma-4b-full+random[2021-03-13/p4/MEK/28]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 76.1% (69.4% to 81.8%) | 0.759 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 70.0% (61.3% to 77.5%) | 0.697 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.883 | 1.000 | 0.0% | no |
| LAS | 60 | 68.3% (55.8% to 78.7%) | 0.683 | 1.000 | 0.0% | no |
| MEK | 60 | 71.7% (59.2% to 81.5%) | 0.712 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.7% | 26.1% | 24.4% | 25.0% | 7.8% |
| Headline (LAS+MEK) | 17.5% | 30.0% | 28.3% | 24.2% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 26.7% | 23.3% |
| LAS | 15.0% | 28.3% | 31.7% | 25.0% | 0.0% |
| MEK | 20.0% | 31.7% | 25.0% | 23.3% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 73.3% | 0.728 | 0.0% |
| 2026-04-18 | 60 | 75.0% | 0.750 | 0.0% |
| 2026-10-18 | 60 | 80.0% | 0.799 | 0.0% |

## Damage

* Perplexity: 20.823 on 75 documents
* Loop rate: 16.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
