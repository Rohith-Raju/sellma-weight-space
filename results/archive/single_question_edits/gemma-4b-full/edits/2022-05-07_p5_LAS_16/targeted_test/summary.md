# Run summary: gemma-4b-full+edit[2022-05-07/p5/LAS/16]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 77.2% (70.6% to 82.7%) | 0.774 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 71.7% (63.0% to 79.0%) | 0.719 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.884 | 1.000 | 0.0% | no |
| LAS | 60 | 70.0% (57.5% to 80.1%) | 0.695 | 1.000 | 0.0% | no |
| MEK | 60 | 73.3% (61.0% to 82.9%) | 0.742 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 21.1% | 24.4% | 23.3% | 23.3% | 7.8% |
| Headline (LAS+MEK) | 24.2% | 27.5% | 26.7% | 21.7% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 26.7% | 23.3% |
| LAS | 20.0% | 26.7% | 30.0% | 23.3% | 0.0% |
| MEK | 28.3% | 28.3% | 23.3% | 20.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 73.3% | 0.729 | 0.0% |
| 2026-04-18 | 60 | 78.3% | 0.788 | 0.0% |
| 2026-10-18 | 60 | 80.0% | 0.804 | 0.0% |

## Damage

* Perplexity: 20.093 on 75 documents
* Loop rate: 20.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
