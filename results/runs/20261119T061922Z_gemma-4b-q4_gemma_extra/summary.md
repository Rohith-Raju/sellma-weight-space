# Run summary: gemma-4b-q4

Backend `llamacpp`, split `gemma_extra`, 120 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 120 | 68.3% (59.6% to 76.0%) | 0.684 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 80 | 60.0% (49.0% to 70.0%) | 0.600 | 1.000 | 0.0% | no |
| ORD | 40 | 85.0% (70.9% to 92.9%) | 0.851 | 1.000 | 0.0% | no |
| LAS | 40 | 57.5% (42.2% to 71.5%) | 0.584 | 1.000 | 0.0% | no |
| MEK | 40 | 62.5% (47.0% to 75.8%) | 0.616 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 19.2% | 30.0% | 18.3% | 26.7% | 5.8% |
| Headline (LAS+MEK) | 20.0% | 37.5% | 17.5% | 25.0% | 0.0% |
| ORD | 17.5% | 15.0% | 20.0% | 30.0% | 17.5% |
| LAS | 22.5% | 30.0% | 12.5% | 35.0% | 0.0% |
| MEK | 17.5% | 45.0% | 22.5% | 15.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2024-10-20 | 60 | 68.3% | 0.681 | 0.0% |
| 2025-04-05 | 60 | 68.3% | 0.686 | 0.0% |

## Damage

* Perplexity: 15.100 on 75 documents
* Loop rate: 20.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
