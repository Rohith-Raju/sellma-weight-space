# Run summary: qwen-9b-full

Backend `hf`, split `dev`, 1350 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 1350 | 88.8% (87.0% to 90.4%) | 0.853 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 900 | 88.4% (86.2% to 90.4%) | 0.847 | 1.000 | 0.0% | no |
| ORD | 450 | 89.6% (86.4% to 92.1%) | 0.866 | 1.000 | 0.0% | no |
| LAS | 450 | 91.1% (88.1% to 93.4%) | 0.879 | 1.000 | 0.0% | no |
| MEK | 450 | 85.8% (82.2% to 88.7%) | 0.815 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 21.0% | 22.4% | 25.2% | 23.6% | 7.8% |
| Headline (LAS+MEK) | 22.2% | 24.2% | 27.9% | 25.7% | 0.0% |
| ORD | 18.7% | 18.9% | 19.8% | 19.3% | 23.3% |
| LAS | 22.7% | 23.8% | 29.1% | 24.4% | 0.0% |
| MEK | 21.8% | 24.7% | 26.7% | 26.9% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 81.7% | 0.812 | 0.0% |
| 2014-04-05 | 60 | 88.3% | 0.847 | 0.0% |
| 2014-10-25 | 60 | 81.7% | 0.788 | 0.0% |
| 2015-03-28 | 60 | 86.7% | 0.820 | 0.0% |
| 2015-10-24 | 60 | 86.7% | 0.824 | 0.0% |
| 2016-04-09 | 60 | 78.3% | 0.777 | 0.0% |
| 2016-10-29 | 60 | 88.3% | 0.836 | 0.0% |
| 2017-04-01 | 60 | 91.7% | 0.881 | 0.0% |
| 2017-10-21 | 60 | 90.0% | 0.854 | 0.0% |
| 2018-04-14 | 30 | 96.7% | 0.914 | 0.0% |
| 2018-10-21 | 60 | 95.0% | 0.880 | 0.0% |
| 2019-04-06 | 60 | 88.3% | 0.827 | 0.0% |
| 2019-10-20 | 60 | 91.7% | 0.897 | 0.0% |
| 2020-10-25 | 60 | 95.0% | 0.929 | 0.0% |
| 2021-03-13 | 60 | 91.7% | 0.900 | 0.0% |
| 2021-05-08 | 60 | 86.7% | 0.845 | 0.0% |
| 2021-10-24 | 60 | 85.0% | 0.829 | 0.0% |
| 2022-03-12 | 60 | 85.0% | 0.842 | 0.0% |
| 2022-05-07 | 60 | 91.7% | 0.855 | 0.0% |
| 2022-10-23 | 60 | 91.7% | 0.891 | 0.0% |
| 2023-03-25 | 60 | 85.0% | 0.801 | 0.0% |
| 2023-10-22 | 60 | 93.3% | 0.880 | 0.0% |
| 2024-04-13 | 60 | 96.7% | 0.932 | 0.0% |

## Damage

* Perplexity: 7.869 on 75 documents
* Loop rate: 88.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
