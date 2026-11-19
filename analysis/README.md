# Analysis scripts and where their results are

Every check run on 2026-11-19 after the main experiments. Scripts that run the model write per-question
results; scripts that only read saved answers write their printed summary to `experiments/analysis/reports/`.

| Script | Runs the model? | What it checks | Output |
|---|---|---|---|
| `rotations.py` | yes | first rotation test on dev (900 LÄS+MEK): answer moved to A, B, C, D | `experiments/analysis/rotation_first_run_dev.jsonl` |
| `rotation_proof.py` | yes | rotation test on dev: unedited three times (determinism), editor 3's change, random control | `experiments/analysis/rotation_proof.json`, `.out` |
| `heldout_position.py` | yes | rotation test, option averaging and a prompt note on the held-out hidden and test papers | `experiments/analysis/heldout_position.json`, `.jsonl` |
| `other_exams.py` | yes | rotation test on Belebele Swedish, Belebele English (the same questions) and MMLU maths; unedited, editor 3, random | `experiments/analysis/other_exams/*.json` (`*_with_choices.json` also records the chosen letter) |
| `letters.py` | no | how often each letter is correct vs chosen (unedited Gemma, dev) | `reports/letters.txt` |
| `habit_q4.py` | no | option-A habit at full precision vs 4-bit (unedited) | `reports/habit_q4.txt` |
| `edited_habit.py` | no | option-A habit for the four trial edits, full precision vs 4-bit, with random controls | `reports/edited_habit.txt` |
| `group_cmp.py` | no | group-trained change vs unedited vs random on hidden and test | `reports/group_cmp.txt` |
| `proof_check.py` | no | determinism of repeat runs and significance of the rotation results | `reports/proof_check.txt` |
| `oe_stats.py` | no | per-position gains and losses, with significance, on the other exams | `reports/oe_stats.txt` |
| `a_picks.py` | no | how often each letter is picked on the other exams, before and after the change | `reports/a_picks.txt` |
| `pass_cmp.py` | no | practice score of every trial edit the AI editors tested (reads `passes/*/request.json`, which is not in git: full GPU copy only) | `reports/pass_cmp.txt` |
| `rotation_choices.py` | yes | rotation test that keeps which option text was chosen in each order (unedited, editor 3; dev and test) | `experiments/analysis/rotation_choices.json` |
| `choice_patterns.py` | no | sorts each question by what Gemma picks across the four orders (knows, habit at A, unsure, wrong belief, position guess, no idea) | `reports/choice_patterns.txt` |
| `edit_sizes.py` | no | how many weights each edit changed | `reports/edit_sizes.txt` |
| `q4_weight_check.py` | yes (llama.cpp) | does 4-bit compression keep editor 3's change? | `reports/q4_weight_check.txt` |
| `q4_rotation.py` | yes (llama.cpp) | rotation test at 4-bit on the hidden and test papers: unedited, editor 3, random control | `experiments/analysis/q4_rotation/*_shard*of4.jsonl` |
| `q4_rotation_report.py` | no | summary of the 4-bit rotation test next to full precision | `reports/q4_rotation_report.txt` |
| `knowledge_gap_vs_training.py` | no | Gemma's never-right questions vs Qwen-9B's answers | `reports/knowledge_gap_vs_training.txt` |

Caveat for `pass_cmp.txt`: it scores each pass in the original option order, so passes where an editor
rotated the options (for example `gemma_b10` pass 5) are miscounted there. Use the editors' own notes
(`editor_runs/<run>/notes/`) for those passes.

Commands and where each script reads and writes on a fresh clone: `REPRODUCE.md`, section 9.
