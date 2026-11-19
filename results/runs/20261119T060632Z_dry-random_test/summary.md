# Run summary: dry-random

Backend `random`, split `test`, 180 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 180 | 22.2% (16.8% to 28.8%) | 0.231 | 0.900 | 0.0% | no |
| Headline (LAS+MEK) | 120 | 22.5% (15.9% to 30.8%) | 0.243 | 0.900 | 0.0% | no |
| ORD | 60 | 21.7% (13.1% to 33.6%) | 0.205 | 0.900 | 0.0% | no |
| LAS | 60 | 20.0% (11.8% to 31.8%) | 0.231 | 0.900 | 0.0% | no |
| MEK | 60 | 25.0% (15.8% to 37.2%) | 0.255 | 0.900 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 24.4% | 18.9% | 28.3% | 18.9% | 9.4% |
| Headline (LAS+MEK) | 25.8% | 18.3% | 33.3% | 22.5% | 0.0% |
| ORD | 21.7% | 20.0% | 18.3% | 11.7% | 28.3% |
| LAS | 20.0% | 20.0% | 40.0% | 20.0% | 0.0% |
| MEK | 31.7% | 16.7% | 26.7% | 25.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2025-10-19 | 60 | 31.7% | 0.269 | 0.0% |
| 2026-04-18 | 60 | 16.7% | 0.209 | 0.0% |
| 2026-10-18 | 60 | 18.3% | 0.214 | 0.0% |

## Damage

* Perplexity: 24.180 on 75 documents
* Loop rate: 0.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
