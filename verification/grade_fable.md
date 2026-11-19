# Independent grading (Fable), 2026-11-19

Graded as the hiring team would, against the brief's five criteria. I read the brief, `RESEARCH_NOTE.md`, `report/report_full.md`, `README.md`, `REPRODUCE.md`, `CONFIG.md`, `NOTICE`, `results/README.md` and `verification/SUMMARY.md`. I spot-checked `config/decision.toml`, the git history, one saved comparison and the editor's `edit.json`. I did not rerun any model and did not check the 2026-dated citations against the web.

## 1. Scores per criterion

| # | Criterion | Score |
|---|---|---|
| 1 | Scientific judgment | 4 / 5 |
| 2 | Experimental design | 4 / 5 |
| 3 | Technical execution | 4 / 5 |
| 4 | Independent thinking | 3 / 5 |
| 5 | Communication and reproducibility | 3 / 5 |
| | **Overall** | **3.5 / 5** |

**1. Scientific judgment: 4.** The submission separates strong from weak evidence better than most. It says the headline is negative ("The hypothesis was wrong"), it marks the four-order test as "exploratory", and it admits "I chose which weight edit to test in depth using its results on the unseen papers" and "I had no positive control". It refuses to rank edit, prompt note and averaging ("None of these differences is statistically clear (exact tests, p ≥ 0.125)"). It loses a point because the recommendation runs well past the evidence. "Gemma's gap in Swedish looks like missing knowledge" rests on one multiple-choice format scored by first-token letter probability, and "distillation during pre-training" is supported only by literature while the one training arm the candidate ran (the group edit) failed.

**2. Experimental design: 4.** The controls do test the claim: a same-size random change (fixed 2 against 83), two no-weight alternatives (prompt note, option averaging), papers published after Gemma's cutoff, 8 hidden older papers, Belebele in Swedish and English to separate "Swedish" from "multiple choice", and a 4-bit rerun. The four-order design and the "same wrong answer on 329 of 335" check are a good way to separate habit from knowledge. Weaknesses: the hypothesis bundles three methods and a compression clause, the informative result is post hoc on an edit selected on the test set, every random control is one seed, and the most obvious alternative for a position bias (subtract a per-letter prior from the logits, as in the cited Zheng et al.) is missing. The single-question rank-1 edit is close to a strawman.

**3. Technical execution: 4.** The pipeline works end to end: deterministic scoring, own Q4_K_M compression of edited and unedited models, pinned model and dataset revisions, dry-run backends (`dry-oracle`, `dry-broken`) and a GPU re-score that matched saved answers within 1e-8 (`verification/SUMMARY.md`). The edit-size accounting (655,360 touched, 366,231 changed after bfloat16 rounding) is careful. Deductions: `git_commit` is `null` in every `run.json`, the loop-rate damage check is admitted to be unusable, the edited GGUFs were deleted, and nearly all code is uncommitted (see criterion 5).

**4. Independent thinking: 3.** The four-order test, the five question kinds and the observation that compression makes the option-A habit worse (252 to 212) are real contributions. But the key discovery (the option-A habit) was made by the AI editors, the candidate does not explain how the editor found the edit or what it does mechanistically, and the brief's central distinction (reading weights against optimising by feedback) gets one sentence in the note and no experimental treatment. The recommendation section is a generic literature survey that any reader could write without this experiment. The report states "I do not read Swedish", which a Swedish-LLM team will notice.

**5. Communication and reproducibility: 3.** Sentences are plain and every number is traceable (`results/README.md` maps claims to files, `CONFIG.md` lists seeds). But the note is about 1,830 words plus a figure, roughly 3.5 pages against a requested "approximately 1–2 pages". `README.md` is an internal bench manual ("ssh sellma-gpu", "Sync from the Mac", "the project owner", "we") that never states the finding and says the test split is 180 questions while the note says 120. The repository has no remote and two commits, with everything else untracked, so the "public GitHub repository" does not exist yet. Reproduction needs a GPU, gated Gemma weights and scraping studera.nu.

## 2. Overall grade and verdict

**Overall: 3.5 / 5. Advance to interview, conditionally.** This is an honest, well-controlled negative result with one useful secondary finding (a 0.0085% edit that reduces a position bias on held-out papers, on other benchmarks and after 4-bit compression, while a random edit does nothing). The candidate's judgment about what is and is not shown is the strongest part, and that is what the brief weights most. It is held back by packaging and focus. The submission is far larger than a 4-hour task (9 verifier agents, 35 references, 25 analysis scripts, a 336-line `REPRODUCE.md`), which makes the "about 4 hours of my own time" line hard to believe and buries the insight. The recommendation is mostly not derived from the experiment. A grader who clones the repo today finds no public repo, a README that reads like a lab notebook and a pre-registration claim that does not exactly match git. With the five fixes below it is a solid 4. Without them a strict grader could put it at 3.

## 3. Five most important weaknesses, with fixes (ordered by impact)

**W1. The repository is not public and not committed.** `git remote -v` is empty. `git log` has two commits. `RESEARCH_NOTE.md`, `report/`, `results/`, `analysis/`, `REPRODUCE.md` and most of `src/` are untracked. This is the first submission requirement.
*Fix:* commit and push to a public GitHub repo, keeping commit `232bdc9` in the history (it is the only proof of the pre-set rule). Remove or ignore `PLAN.md`, `docs/EDITOR_WORKBENCH.md` and the stale figures `fig1`–`fig4`, `fig6` unless they are referenced. Clone the public repo fresh and run `make test` once.

**W2. The pre-registration claim does not exactly match git, and the README contradicts the note.** The note says the rule was "fixed before any model was run (`config/decision.toml`, committed to git about six hours before the first run)". The committed file has no `headline_sections = ["LAS", "MEK"]` line. That line is an uncommitted edit (file time 01:55, after the 01:03 commit, before the first run at 06:06 UTC). `README.md` still says "rule 1 uses all questions together, ORD included" and "180 test questions". The "kept at 4-bit" clause is in `PLAN.md`, not in `decision.toml`.
*Fix:* add one sentence to note section 2 and report section 2: "The committed rule scored all 180 test questions. Before the first run I narrowed it to the 120 reading and sentence-completion questions. That change is not in a commit. The verdict is the same on all 180: change +0.006, 95% interval −0.011 to +0.026." (Numbers from `results/experiments/editor/trial_b50_k/comparisons/full_edited_vs_baseline_test/test__vs__test/comparison.md`.) Correct the two README lines.

**W3. The note is too long and the submission is over-scoped for "4 hours".** The note is about 1,830 words. Section 6 (recommendation) and section 5 repeat literature that is in the report. The checklist of seven experiment steps and two models dilutes the single claim the brief asks for.
*Fix:* cut the note to about 900 words. Drop Qwen and the one-question edits to one sentence. Cut section 6 to three lines ("Do not pursue weight edits to add Swedish knowledge. Consider them only as a cheap fix for a known behaviour. Next experiment: ..."). Move the Llama/Gemma 2 numbers and distillation detail to the report only. In AI usage, replace "About 4 hours of my own time" with an honest split, for example: "About 4 hours of my own decisions and review. Agents ran for about 17 hours of wall-clock time, mostly unattended."

**W4. The report never says how the edit was found or what it does, which is the brief's central question.** The brief says "Pay particular attention to the distinction between reasoning directly about weights and optimizing edits through repeated model feedback." The saved `edit.json` says the edit is "layer-23 k_proj rows of heads 2 and 3 raising attention to the first option newline". That one line is the most interesting mechanistic fact in the project and it is absent from both documents. The reader also cannot tell whether the editor read weights, read activations or used trial and error.
*Fix:* add three sentences to report section 4 and one to the note: "The 50-run editor did not find this by reading weights alone. It read attention patterns on practice questions, saw that two attention heads in layer 23 paid little attention to the first option, and changed the key weights of those heads so they attend to it more. It then tested four variants with model runs. So this is the article's feedback route, not its weights-only route." (Verify the wording against `results/editor_runs/gemma_b50/notes/` before using it.)

**W5. The recommendation outruns the evidence, and the obvious baseline for the positive finding is missing.** "Changing Gemma's weights changed how it picks an answer, not what it knows" is shown for one edit that targets option position. It cannot support "Do not use weight edits to make a model better at Swedish" in general, and "Train the knowledge in: distillation during pre-training" is untested here. Separately, the edit amounts to a push toward option A (picks of A when wrong rise from 5.0% to 8.0%), and the cheapest comparison, adding a constant to the option-A logit or subtracting a per-letter prior estimated on the older papers, was not run. Meanwhile the one practically useful result (compression worsens the habit and the edit still fixes 48 and breaks 0 at 4-bit) is undersold.
*Fix:* reword the headline recommendation to: "In my tests, weight edits did not add Swedish knowledge. I would not investigate them for that purpose. They may be worth a small follow-up as a fix for a known behaviour that must survive 4-bit compression. Training routes below come from the literature, not from my experiment." Run the logit-offset baseline on the saved probabilities (no GPU needed: `heldout_position.jsonl` has per-letter probabilities) and report it next to the prompt note and averaging. If there is no time, list it as a limitation: "I did not compare with a simple per-letter probability correction."

Smaller points a grader may notice:
- Twelve references are listed but never cited in the body (Adelani, Bandarkar, Dang, Hendrycks, Samuel, Shi, Tang, Wang, both Zhang SnakModel and Kunz translationese, Ollama, Qwen Team). Delete them or cite them. Belebele and MMLU should be cited where they are used.
- "about 4 seconds for ROME" is linked to the MEMIT paper (arXiv 2210.07229, "ICLR 2023"). ROME is 2202.05262.
- "For up to 10 questions each model got wrong" but "none of the 12" edits: say "5 Gemma and 7 Qwen edits".
- The note says the hypothesis needs the edit to "get no worse at ordinary Swedish text", then reports the looping check "was too noisy to use". The saved comparison shows rule 3b passed (−2.0 points). State that it passed but is not relied on.
- Figure files are named `fig5`, `fig9`, `fig7`, `fig8` but appear as Figures 1–4.
- The report says the editors "noticed" the habit but drops the supporting test (124 against 9, p ≈ 5e-27) that `verification/SUMMARY.md` has.
- "Nine independent AI agents" are the same model family as the author. Say "separate" instead of "independent".
- The two Swedish examples were translated by an AI agent and the author cannot read them. Have a Swedish speaker check them, or say so next to the examples, not only in section 8.
- Eight citations are dated 2026. I could not verify them offline. One wrong reference would cost a lot of trust, so re-open each link once before submitting.

## 4. Requirement checklist

| Item | Status | Comment |
|---|---|---|
| Public GitHub repository | **Missing** | No remote. Two commits. Almost everything untracked. |
| Code and a straightforward way to reproduce the main experiment | **Partly** | `make main` exists and `scripts/check_main.py` compares against saved answers. It needs a GPU, gated Gemma weights and `make data` (scrape of studera.nu). `REPRODUCE.md` is 336 lines, which is not "straightforward". The no-GPU path (`make test`, `make reports`, `make verify`) is good. Code is not committed. |
| Results: evaluation examples | **Met** | Two quoted items in the report, per-question `answers.jsonl` for every run. The note itself has no example. |
| Results: outcomes | **Met** | `results/README.md` maps each claim to a file. |
| Results: configuration, model revisions | **Met** | Revisions in `NOTICE` and `CONFIG.md`. Weight checksums in `run.json`. Gap: `git_commit` is `null` in every run. |
| Results: random seeds | **Met** | `CONFIG.md` section 2. Each control has one seed only. |
| Research note, about 1–2 pages, Markdown | **Partly** | All five required parts are present. About 1,830 words plus a figure, roughly 3.5 pages. |
| PDF report based on the note | **Met** | `report/report.pdf`, 8 pages, same structure as the note. Rebuild it after any text fix. |
| AI usage description | **Met** | Note section 8 and report section 8. Candid ("I do not read Swedish"). The "4 hours" figure needs the wall-clock context. |
| Data and weight rights | **Met** | Strong: exam texts kept out of the repo, `NOTICE` with Gemma terms, Apache notice for Qwen, CC BY-SA for Belebele, Riksdag attribution, `report/DATA_RIGHTS.md`. No licence file for the candidate's own code. |
| Task 1: critical examination, weights-only against feedback | **Partly** | The critique of the article is sharp (45.8% with thinking off). The weights-only against feedback distinction gets one sentence and is not tied to the candidate's own editors (W4). |
| Task 2: hypothesis, why it matters, support, challenge | **Met** | Report section 2 has all four. |
| Task 4: baseline, random edit, prompting, conventional method, held-out split, regression check | **Met** | All present. The conventional method (low-rank training on 205 wrong answers) is weak and gets two lines. |
| Task 5: recommendation and next experiment | **Met** | Clear, but over-reaching (W5). |

## 5. Three strongest parts

1. **The four-order test with the "same wrong answer" check.** Asking each question with the right answer at A, B, C and D turns a 1-point accuracy change into an interpretable result: 83 fixed and 0 broken at option A, none of the 109 always-wrong questions became fully right, and the edited model kept the identical wrong answer on 329 of 335 (110 of 110 and 40 of 40 on unseen papers). This is the evidence that separates "habit removed" from "knowledge added", which is exactly the fourth example question in the brief.
2. **Honest handling of selection and of the pre-set rule.** The candidate reports the best edit as "no improvement" by the pre-set rule (change +0.009, 95% interval −0.016 to +0.038) even though a headline of "83 fixed, 0 broken" was available, flags the test-set selection, and shows that averaging over orders with no weight change does as well (75.0% against 70.8% on the test papers).
3. **The reading of the article and the compression result.** The critique finds the decisive number in the source (unedited model with thinking off already at 45.8%, so the edit removed a failure). The 4-bit rerun with a matched random control is a non-trivial extra: compression makes the habit worse (252 to 212 right at option A), and the edit still fixes 48 and breaks 0, with 86% of the changed weights surviving rounding.
