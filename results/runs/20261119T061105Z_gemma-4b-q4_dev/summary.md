# Run summary: gemma-4b-q4

Backend `llamacpp`, split `dev`, 1350 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 1350 | 70.6% (68.1% to 73.0%) | 0.702 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 900 | 65.7% (62.5% to 68.7%) | 0.651 | 1.000 | 0.0% | no |
| ORD | 450 | 80.4% (76.5% to 83.8%) | 0.804 | 1.000 | 0.0% | no |
| LAS | 450 | 67.6% (63.1% to 71.7%) | 0.668 | 1.000 | 0.0% | no |
| MEK | 450 | 63.8% (59.2% to 68.1%) | 0.634 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 14.2% | 26.1% | 29.6% | 22.8% | 7.2% |
| Headline (LAS+MEK) | 13.1% | 29.1% | 33.4% | 24.3% | 0.0% |
| ORD | 16.4% | 20.2% | 22.0% | 19.8% | 21.6% |
| LAS | 14.4% | 23.3% | 37.3% | 24.9% | 0.0% |
| MEK | 11.8% | 34.9% | 29.6% | 23.8% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 71.7% | 0.707 | 0.0% |
| 2014-04-05 | 60 | 65.0% | 0.650 | 0.0% |
| 2014-10-25 | 60 | 60.0% | 0.611 | 0.0% |
| 2015-03-28 | 60 | 73.3% | 0.729 | 0.0% |
| 2015-10-24 | 60 | 66.7% | 0.654 | 0.0% |
| 2016-04-09 | 60 | 73.3% | 0.725 | 0.0% |
| 2016-10-29 | 60 | 76.7% | 0.761 | 0.0% |
| 2017-04-01 | 60 | 66.7% | 0.664 | 0.0% |
| 2017-10-21 | 60 | 66.7% | 0.660 | 0.0% |
| 2018-04-14 | 30 | 70.0% | 0.671 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.828 | 0.0% |
| 2019-04-06 | 60 | 68.3% | 0.684 | 0.0% |
| 2019-10-20 | 60 | 75.0% | 0.746 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.763 | 0.0% |
| 2021-03-13 | 60 | 71.7% | 0.702 | 0.0% |
| 2021-05-08 | 60 | 63.3% | 0.634 | 0.0% |
| 2021-10-24 | 60 | 60.0% | 0.598 | 0.0% |
| 2022-03-12 | 60 | 70.0% | 0.699 | 0.0% |
| 2022-05-07 | 60 | 81.7% | 0.813 | 0.0% |
| 2022-10-23 | 60 | 75.0% | 0.750 | 0.0% |
| 2023-03-25 | 60 | 61.7% | 0.616 | 0.0% |
| 2023-10-22 | 60 | 68.3% | 0.684 | 0.0% |
| 2024-04-13 | 60 | 78.3% | 0.784 | 0.0% |

## Damage

* Perplexity: 15.100 on 75 documents
* Loop rate: 20.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
