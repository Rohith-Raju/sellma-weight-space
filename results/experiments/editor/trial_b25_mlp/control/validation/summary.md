# Run summary: gemma-4b-full+editor[trial_b25_mlp:random]

Backend `hf`, split `dev`, 150 questions.

## Exam score

The headline (LAS+MEK) is the number that counts for the decision. ORD is mostly vocabulary (knowledge), so it is shown separately.

| Section | n | Accuracy (95% CI) | Mean P(correct) | Mean P(letters) | Unsure | At ceiling |
|---|---|---|---|---|---|---|
| All | 150 | 71.3% (63.6% to 78.0%) | 0.716 | 1.000 | 0.0% | no |
| Headline (LAS+MEK) | 100 | 67.0% (57.3% to 75.4%) | 0.673 | 1.000 | 0.0% | no |
| ORD | 50 | 80.0% (67.0% to 88.8%) | 0.801 | 1.000 | 0.0% | no |
| LAS | 50 | 66.0% (52.2% to 77.6%) | 0.660 | 1.000 | 0.0% | no |
| MEK | 50 | 68.0% (54.2% to 79.2%) | 0.686 | 1.000 | 0.0% | no |

## Chosen letters

Share of answers where each letter was the most likely choice.

| Group | A | B | C | D | E |
|---|---|---|---|---|---|
| All | 20.7% | 23.3% | 25.3% | 25.3% | 5.3% |
| Headline (LAS+MEK) | 19.0% | 25.0% | 25.0% | 31.0% | 0.0% |
| ORD | 24.0% | 20.0% | 26.0% | 14.0% | 16.0% |
| LAS | 16.0% | 18.0% | 30.0% | 36.0% | 0.0% |
| MEK | 22.0% | 32.0% | 20.0% | 26.0% | 0.0% |

## By exam date

| Exam date | n | Accuracy | Mean P(correct) | Unsure |
|---|---|---|---|---|
| 2018-04-14 | 30 | 73.3% | 0.733 | 0.0% |
| 2022-03-12 | 60 | 71.7% | 0.717 | 0.0% |
| 2023-10-22 | 60 | 70.0% | 0.706 | 0.0% |

## Damage

No damage.json for this run (it was run with --skip-damage, or the checks did not finish).
