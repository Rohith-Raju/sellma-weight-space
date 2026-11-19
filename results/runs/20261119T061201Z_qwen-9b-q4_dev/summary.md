# Run summary: qwen-9b-q4

Backend `llamacpp`, split `dev`, 1350 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 1350 | 87.4% (85.5% to 89.1%) | 0.831 | 0.999 | 0.0% | no |
| Headline (LAS+MEK) | 900 | 87.1% (84.8% to 89.1%) | 0.826 | 0.999 | 0.0% | no |
| ORD | 450 | 88.0% (84.7% to 90.7%) | 0.841 | 0.999 | 0.0% | no |
| LAS | 450 | 90.0% (86.9% to 92.4%) | 0.864 | 0.999 | 0.0% | no |
| MEK | 450 | 84.2% (80.6% to 87.3%) | 0.788 | 0.999 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.8% | 21.8% | 26.0% | 24.0% | 7.4% |
| Headline (LAS+MEK) | 22.1% | 23.8% | 28.1% | 26.0% | 0.0% |
| ORD | 18.2% | 17.8% | 21.8% | 20.0% | 22.2% |
| LAS | 23.3% | 23.8% | 29.6% | 23.3% | 0.0% |
| MEK | 20.9% | 23.8% | 26.7% | 28.7% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2013-10-26 | 60 | 83.3% | 0.808 | 0.0% |
| 2014-04-05 | 60 | 85.0% | 0.809 | 0.0% |
| 2014-10-25 | 60 | 83.3% | 0.774 | 0.0% |
| 2015-03-28 | 60 | 83.3% | 0.783 | 0.0% |
| 2015-10-24 | 60 | 85.0% | 0.801 | 0.0% |
| 2016-04-09 | 60 | 80.0% | 0.784 | 0.0% |
| 2016-10-29 | 60 | 86.7% | 0.805 | 0.0% |
| 2017-04-01 | 60 | 86.7% | 0.838 | 0.0% |
| 2017-10-21 | 60 | 88.3% | 0.825 | 0.0% |
| 2018-04-14 | 30 | 96.7% | 0.908 | 0.0% |
| 2018-10-21 | 60 | 91.7% | 0.825 | 0.0% |
| 2019-04-06 | 60 | 83.3% | 0.797 | 0.0% |
| 2019-10-20 | 60 | 86.7% | 0.845 | 0.0% |
| 2020-10-25 | 60 | 95.0% | 0.923 | 0.0% |
| 2021-03-13 | 60 | 93.3% | 0.868 | 0.0% |
| 2021-05-08 | 60 | 88.3% | 0.837 | 0.0% |
| 2021-10-24 | 60 | 86.7% | 0.839 | 0.0% |
| 2022-03-12 | 60 | 86.7% | 0.851 | 0.0% |
| 2022-05-07 | 60 | 90.0% | 0.850 | 0.0% |
| 2022-10-23 | 60 | 90.0% | 0.867 | 0.0% |
| 2023-03-25 | 60 | 78.3% | 0.758 | 0.0% |
| 2023-10-22 | 60 | 91.7% | 0.863 | 0.0% |
| 2024-04-13 | 60 | 95.0% | 0.899 | 0.0% |

## Damage

* Perplexity: 7.768 on 75 documents
* Loop rate: 100.0% of 50 prompts
* Loop rule: a reply loops if some run of 4 consecutive words (lowercased, punctuation removed) occurs 3 or more times in the reply; greedy continuation, max 200 new tokens
