# Run summary: gemma-4b-full+edits[combined 5]

Backend `hf`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 75.0% (68.2% to 80.8%) | 0.734 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 68.3% (59.6% to 76.0%) | 0.660 | 1.000 | 0.0% | no |
| ORD | 60 | 88.3% (77.8% to 94.2%) | 0.881 | 1.000 | 0.0% | no |
| LAS | 60 | 66.7% (54.1% to 77.3%) | 0.646 | 1.000 | 0.0% | no |
| MEK | 60 | 70.0% (57.5% to 80.1%) | 0.674 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 17.2% | 20.0% | 24.4% | 30.6% | 7.8% |
| Headline (LAS+MEK) | 18.3% | 20.8% | 28.3% | 32.5% | 0.0% |
| ORD | 15.0% | 18.3% | 16.7% | 26.7% | 23.3% |
| LAS | 15.0% | 18.3% | 33.3% | 33.3% | 0.0% |
| MEK | 21.7% | 23.3% | 23.3% | 31.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 66.7% | 0.671 | 0.0% |
| 2026-04-18 | 60 | 76.7% | 0.746 | 0.0% |
| 2026-10-18 | 60 | 81.7% | 0.784 | 0.0% |

## Damage

* Perplexity: 19.444 on 75 documents
* Loop rate: 24.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
