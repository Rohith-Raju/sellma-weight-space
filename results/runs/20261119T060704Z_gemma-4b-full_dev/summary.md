# Run summary: gemma-4b-full

Backend `hf`, split `dev`, 1350 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 1350 | 73.1% (70.7% to 75.4%) | 0.730 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 900 | 68.3% (65.2% to 71.3%) | 0.682 | 1.000 | 0.0% | no |
| ORD | 450 | 82.7% (78.9% to 85.9%) | 0.827 | 1.000 | 0.0% | no |
| LAS | 450 | 71.6% (67.2% to 75.5%) | 0.713 | 1.000 | 0.0% | no |
| MEK | 450 | 65.1% (60.6% to 69.4%) | 0.651 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 16.4% | 25.5% | 25.9% | 25.4% | 6.7% |
| Headline (LAS+MEK) | 15.6% | 28.0% | 28.1% | 28.3% | 0.0% |
| ORD | 18.2% | 20.4% | 21.6% | 19.6% | 20.2% |
| LAS | 15.3% | 25.8% | 31.6% | 27.3% | 0.0% |
| MEK | 15.8% | 30.2% | 24.7% | 29.3% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 73.3% | 0.734 | 0.0% |
| 2014-04-05 | 60 | 66.7% | 0.666 | 0.0% |
| 2014-10-25 | 60 | 66.7% | 0.672 | 0.0% |
| 2015-03-28 | 60 | 75.0% | 0.752 | 0.0% |
| 2015-10-24 | 60 | 60.0% | 0.582 | 0.0% |
| 2016-04-09 | 60 | 70.0% | 0.707 | 0.0% |
| 2016-10-29 | 60 | 80.0% | 0.788 | 0.0% |
| 2017-04-01 | 60 | 75.0% | 0.755 | 0.0% |
| 2017-10-21 | 60 | 76.7% | 0.770 | 0.0% |
| 2018-04-14 | 30 | 76.7% | 0.759 | 0.0% |
| 2018-10-21 | 60 | 83.3% | 0.833 | 0.0% |
| 2019-04-06 | 60 | 71.7% | 0.718 | 0.0% |
| 2019-10-20 | 60 | 73.3% | 0.728 | 0.0% |
| 2020-10-25 | 60 | 76.7% | 0.769 | 0.0% |
| 2021-03-13 | 60 | 71.7% | 0.718 | 0.0% |
| 2021-05-08 | 60 | 70.0% | 0.708 | 0.0% |
| 2021-10-24 | 60 | 58.3% | 0.575 | 0.0% |
| 2022-03-12 | 60 | 71.7% | 0.718 | 0.0% |
| 2022-05-07 | 60 | 86.7% | 0.866 | 0.0% |
| 2022-10-23 | 60 | 80.0% | 0.788 | 0.0% |
| 2023-03-25 | 60 | 66.7% | 0.672 | 0.0% |
| 2023-10-22 | 60 | 71.7% | 0.717 | 0.0% |
| 2024-04-13 | 60 | 81.7% | 0.814 | 0.0% |

## Damage

* Perplexity: 20.828 on 75 documents
* Loop rate: 10.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
