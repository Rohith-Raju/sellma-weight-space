# Independent grade: Weight-Space Programming for SE.LLMA

Grader: Claude Opus 5.5, acting as the hiring team. Date: 2026-11-19.
Read: the brief (3 pages), `RESEARCH_NOTE.md`, `report/report_full.md` (the PDF build log says 8 pages), `README.md`, `REPRODUCE.md`, `CONFIG.md`, `NOTICE`, `results/README.md`, `verification/SUMMARY.md`, `verification/review_challenge.md`, `PLAN.md`.
Spot-checked: `git log`, `git diff config/decision.toml`, `results/experiments/editor/trial_b50_k/results.md`, `results/experiments/analysis/reports/a_habit_all_tests.txt`, `oe_stats.txt`, `group_cmp.txt`, `results/editor_runs/trial_b50_k/final_edit.*` and `results/experiments/analysis/rotation_choices.json`.

---

## 1. Scores by criterion

| Criterion | Score |
|---|---|
| 1. Scientific judgment | **4** |
| 2. Experimental design | **3** |
| 3. Technical execution | **4** |
| 4. Independent thinking | **3** |
| 5. Communication and reproducibility | **4** |

**1. Scientific judgment: 4.** The candidate rejects their own hypothesis under a rule set in advance. They print the verdict ("change +0.009, 95% interval −0.016 to +0.038", report section 4). They label the four-order test, the five kinds and the compression measure as exploratory. They also admit that the edit was chosen "using its results on the unseen papers" (note section 7). That is the honesty the brief asks for. The score is not a 5 because the bridge from the experiment to the recommendation is not supported. "Gemma's gap in Swedish looks like missing knowledge" (note section 5) rests on the 109 never-right questions. But 386 of 900 questions are order-sensitive, and averaging over orders gave the largest test gain (70.0% → 75.0%, report section 4). Also, "It also shows in English, so it is not about Swedish" (note section 4) goes past the candidate's own data. On the same 500 Belebele questions, Gemma is wrong only at option A on 31 questions in Swedish and 15 in English (`a_habit_all_tests.txt`).

**2. Experimental design: 3.** The controls are good. There is a test split from after the model's training cutoff, a random change with the same norm on the same rows, a prompt change, option averaging, a small fine-tune, a perplexity check, two other exams and 4-bit compression (report section 3). But the hypothesis was close to a guaranteed null. One-question rank-1 edits push a letter for one prompt (report section 3, step 2), and the article already concludes that edits "could not add an ability the model lacked" (note section 1). The one positive result (the option-A edit) was found after the fact and chosen on the held-out papers. The AI editors used 5, 7 and 11 of their 10, 25 and 50 runs (note section 3), so the article's feedback-heavy condition was never really tested. The fine-tuning alternative was weak: 205 multiple-choice answers including ORD, and step 50 won only "on the tie-break" (`verification/SUMMARY.md`, point 5).

**3. Technical execution: 4.** The work runs and is checked. Scoring is deterministic and three runs agreed on all 3,600 answers (report section 7). An independent GPU re-score matched every sampled answer (`verification/SUMMARY.md`). I confirmed that the shipped edit is rank-1 (singular values 0.302, then 4e-9) and that `rotation_choices.json` reproduces the 2,450 and 2,486 totals in report section 4. The score is not a 5 because of provenance gaps. `git_commit` is `null` in every `run.json` (`CONFIG.md`). The Hugging Face revisions were inferred later, not recorded at download (`CONFIG.md`). The edited 4-bit GGUFs were deleted (`verification/SUMMARY.md`). The Qwen single-question experiment was stopped at 7 of 10.

**4. Independent thinking: 3.** The four-order test is a good idea. It splits "knows it but avoids A" from "always wrong" and gives clean paired counts (83/0, 27/0, 11/0). The five-kinds view and the 4-bit survival check are also thoughtful. But the candidate did not use the brief's best opening, the article's "obstacle vs ability" idea, on a Swedish-specific obstacle. Examples are answering in English, code-switching, or translationese, and the reference list even includes a Swedish translationese dataset (Kunz, Jarochenko, Bollmann 2026). The mechanism of the edit is in the repo but not in the note or the report: "layer-23 k_proj rows of heads 2 and 3 raising attention to the first option newline" (`results/editor_runs/trial_b50_k/final_edit.json`). The distillation recommendation is standard advice from the literature, not something the experiment found.

**5. Communication and reproducibility: 4.** Reproducibility is unusually strong. `REPRODUCE.md` has make targets, `CONFIG.md` has checksums and seeds, `results/README.md` maps claims to files, `NOTICE` covers the licences, and `make reports` rebuilds 8 reports byte for byte. Communication is weaker. The note is about 1,800 words plus a figure, against "approximately 1–2 pages". It is dense with counts, and the reader must keep "fixed", "broken", "kinds 4 and 5" and three test sets in their head. `README.md` is an internal bench manual: it says "`ssh sellma-gpu`", it never states the finding, and it contradicts the note (see weakness 3). There is no public repository yet. `git log` shows only the skeleton commit, and almost all files are untracked.

---

## 2. Overall grade: **4 / 5** (weak 4)

**Verdict.** Advance to the next round. This is a careful negative result with real controls, a pre-set decision rule, honest exploratory labels and a level of traceability that few candidates reach. The four-order analysis shows clearly that the AI-found edit shifts a letter preference and does not change which answers Gemma believes (329 of 335 wrong picks unchanged). The interview should probe three things. First, why this hypothesis, which the article had already predicted would fail? Second, does the "missing knowledge" conclusion hold when order-sensitivity is the larger share of errors? Third, how much of the work and of the judgment came from the candidate rather than from the agents? The volume of work (an 8-page report, 30+ references, nine verification agents, a 336-line `REPRODUCE.md`) is far beyond the 4-hour target. Some reviewers will read this as good scoping with AI. Others will read it as an inability to stop. The note should own this in one line.

---

## 3. The five most important weaknesses (ordered by impact on the grade)

### W1. The conclusion and the recommendation go past the evidence

**Where:** note section 5, "Its gap in Swedish looks like missing knowledge. My results and other research point to training as the way to add it." Note section 6, "Do not use weight edits to make a model better at Swedish. Train the knowledge in". Both are repeated in report sections 5 and 6.

**Why it costs points:** this hits criterion 1 directly.
- Order-sensitive questions (262 "unsure" plus 124 "avoids A") outnumber never-right questions (109) by more than 3 to 1 (report section 4 and `review_challenge.md` C1).
- `README.md` says the exam "tests language skill, not knowledge of facts", so "knowledge" is undefined and conflicts with the repo's own framing.
- No positive control was run (note section 7).
- The candidate's own citation shows Nordic continued pre-training did not help on this exam: 62.4% against 63.0% (note section 6, point 3).

**Fix (exact wording, note section 5):** "Changing Gemma's weights changed how it picks an answer. I found no sign that it changed which answers Gemma can reach. 109 of 900 questions were wrong in every order, and no edit reduced that group. A larger group, 386 questions, flips with the order of the options. I did not test whether training helps either group." In section 6, replace "Do not use weight edits…" with "I found no evidence that weight edits improve Swedish ability, and I do not recommend them for that. Whether distillation helps is untested here." Also change "No study has yet tested distillation for Swedish" to "I found no study that tests distillation for Swedish."

### W2. The design could hardly have supported the hypothesis, and the obvious one-number baseline for the positive result is missing

**Where:** note sections 2 and 3, report section 3, steps 2, 5 and 6.

**Why it costs points:** criteria 2 and 4. The hypothesis tests whether edits built on exam mistakes add generalisable ability, which the article already says they cannot. The one positive result is a position bias that the brief's "inference settings" bullet could fix with a single number. I checked this offline from the saved `results/experiments/analysis/rotation_choices.json`, with no model run. I multiplied the probability of whatever option sits in position A by a constant, tuned on dev over a coarse grid (best about 1e5). That gives 2,491 of 3,600 on dev, against 2,486 for the edit. It gives 339 of 480 on the test papers, against 336 for the edit and 335 unedited. Right-at-A on test is 83, against 80 for the edit. So a 1-parameter letter bias matches the 2,816-parameter weight edit. This is my own rough check and it is not in the submission.

**Fix:** add this baseline in report section 4 and note section 4, as one more row next to the prompt note and averaging: "A single bias toward option A, tuned on the older papers, did as well as the weight edit (test: 339 vs 336 of 480 answers)." In note section 7, add: "My first two edit methods push one answer letter for one question. Their failure was expected from the article's own conclusion. A fairer test of 'edits remove obstacles' would target a Swedish-specific obstacle, such as replies in English to Swedish prompts." Also say plainly in section 3: "The AI editors stopped after 5, 7 and 11 runs, so this is not a replication of the article's 50-round condition."

### W3. The pre-registration claim cannot be checked in git and contradicts the README

**Where:** note section 2: "fixed before any model was run (`config/decision.toml`, committed to git about six hours before the first run)".

**Facts:**
- `git show 232bdc9:config/decision.toml` has no `headline_sections`. The working tree adds `headline_sections = ["LAS", "MEK"]`, and that change is not committed (`git diff`).
- `README.md` line 99 still says "rule 1 uses all questions together, ORD included". It also gives the test split as 180 questions (README split table), while the note uses 120.
- `PLAN.md` records "Headline score = LÄS + MEK" at "~02:00 UTC, before any real model was run", but `PLAN.md` is untracked, so its date is not verifiable.

**Why it costs points:** a grader who checks the one git-verifiable claim of pre-registration finds a mismatch. That undermines criterion 1 even if the intent was honest.

**Fix:** commit everything now. In note section 2, write: "Rules 1–3 were committed at 00:03 UTC (commit 232bdc9). The choice to score only LÄS + MEK was written in `PLAN.md` at about 02:00 UTC, before the first model run at 06:07 UTC, and added to `decision.toml` later." Delete the README sentence "Note that rule 1 uses all questions together, ORD included" and change "180" to "180 (120 LÄS + MEK, which decide the verdict)".

### W4. Selective or inconsistent reporting between the note, the report and the repo

These are each small, but together they make a reader distrust the summary.
- **"Not about Swedish."** The habit is about twice as strong in Swedish as in English on the same Belebele questions: 31 vs 15 wrong-only-at-A, and A accuracy 76.4% vs 83.4% (`a_habit_all_tests.txt`). Fix: "The habit also appears in English, but it is about twice as strong in Swedish on the same questions. It may grow with uncertainty."
- **gemma_extra omitted.** On the two post-cutoff 2024–25 papers, the option-A edit changed accuracy by −2.5 points (95% CI −5.8 to +0.0) (`results/experiments/editor/trial_b50_k/results.md`). Neither document mentions it (`review_challenge.md` m13 flagged this and it is still open). Fix: add one sentence to report section 4.
- **The article's re-run.** The note says the re-run "got the same result" and gives the +4.3 points (edited with thinking on vs unedited with thinking off). The report says "similar (7.4% to 50.2%)" and gives only −2.2 (thinking off). The +4.3 has a 95% interval of 1.9 to 6.8 (`review_challenge.md` M8), so "fixed a behaviour, not maths ability" is slightly too strong. Fix: use "similar" in both, and add to report section 1: "With thinking on, the edited model still beat the unedited model with thinking off by 4.3 points (95% interval 1.9 to 6.8). Most, not all, of the gain was already available."
- **The summary headline.** "that fix survives compression to 4-bit" (summary in both documents) has no "exploratory" tag, while section 4 has one. Fix: "…and in an exploratory check that fix survived compression to 4-bit."
- **Naming.** The note says "nine independent AI agents" (section 3, step 7), while the report says "Separate AI agents". Use "separate" in both, since they share a model family.
- **The mechanism.** The edit raises attention from heads 2–3 of layer 23 to the first option line (`final_edit.json`). Neither document says this. It is the most direct answer to the brief's "reasoning about weights vs feedback" point. Fix: one sentence in report section 4 saying how editor 3 found it (attention scores plus a few model runs, not weights alone).

### W5. Packaging: no public repository, an internal README, and an over-length note

**Facts:**
- There is no `git remote`. `git status` shows 41 modified or untracked paths, including all of `src/bench/*.py`, `results/`, `report/` and the note.
- `README.md` opens as a bench manual and includes "Connect to the GPU machine… `ssh sellma-gpu`" and "owner decision". `PLAN.md` says "Defaults chosen while the owner was asleep".
- The note is about 1,800 words against "approximately 1–2 pages".

**Fix:**
1. Commit in a few logical commits and push to a public GitHub repository. Check it from a fresh clone with `make test reports figures`.
2. Put a 6-line "Result" block at the top of `README.md`: question, verdict, the 83/0 · 27/0 · 11/0 counts, the "behaviour, not knowledge" line, the recommendation, and links. Move "How to use it" (ssh, tmux) to `docs/`.
3. Cut the note to about 1,100 words. Shorten section 1 to three sentences, merge sections 7 and 8, and drop the duplicate "Report, section N" pointers.
4. Add one line to the AI-usage section: "Wall-clock about 17 hours (skeleton commit 01:03 to final report 18:13, local time), mostly unattended agent runs, on one rented RTX PRO 6000."

---

## 4. Requirement checklist

| Requirement (brief, "Submission") | Status | Evidence |
|---|---|---|
| Public GitHub repository | **Missing** | No remote. Only 2 commits (skeleton). Nearly all work is untracked (`git status`). |
| Code + a straightforward way to reproduce the main experiment | **Met** | `make main` (`REPRODUCE.md` section 5), with expected outputs and `scripts/check_main.py`. A GPU and gated Gemma access are needed. The code is not yet committed. |
| Results: evaluation examples | **Met** | Two worked examples with translations (report section 4). Per-question answers are in `results/`. The question texts are rebuilt with `make data`. |
| Results: outcomes | **Met** | Report section 4, `results/`, claim-to-file map in `results/README.md`. |
| Results: configuration details | **Met** | `CONFIG.md` covers inference, decision rule and group-training settings. |
| Results: model revisions | **Partly** | Revisions are given, but they were "from the reference check… not from the download itself". Shard sha256 are the binding identity (`CONFIG.md`). |
| Results: random seeds | **Met** | `CONFIG.md` section 2. The AI-editor agent seeds are "not recorded", which is stated. |
| Research note, about 1–2 pages, Markdown | **Partly** | It covers hypothesis, experiment, results, interpretation and recommendation. At about 1,800 words plus a figure it is over length. |
| PDF report based on the note | **Met** | `report/report.pdf`, 8 pages, built from `report_full.md` at the same time (18:13). |
| AI usage | **Met** | Note section 8 and report section 8. Wall-clock time and agent hours are not given. |
| Data and weights you have the right to use and share | **Met** | Exam texts are withheld and quoted only under 22 § URL. `NOTICE` covers Gemma terms, Apache-2.0, CC BY-SA, MIT and Riksdagen attribution. |
| Brief task 1: strongest evidence, limits, justified/unjustified conclusions, weights vs feedback | **Partly** | Report section 1 covers "What it shows / does not show" and the role of feedback. The note never names "strongest evidence" or "not justified" explicitly. |
| Brief task 4: baseline, random edit, inference settings, fine-tuning or steering, separate eval data, at least one regression | **Met, with caveats** | All are present. The fine-tune is weak and there is no steering. The edit was selected using held-out papers. |
| Brief task 5: recommendation + next experiment | **Met** | Note section 6. The recommendation leans on literature (see W1). |

---

## 5. The three strongest parts

1. **The four-order (rotation) test with paired fixed/broken counts.** It turns a vague "+1 point" into a clean mechanism statement: 83 fixed and 0 broken at option A on dev, 27/0 on hidden papers, 11/0 on test, while a random change of the same norm fixes 2 (report section 4). It shows the same one-sided pattern on Belebele Swedish (21/0), Belebele English (12/0) and MMLU (45/0) (`oe_stats.txt`). The "same wrong answer on 329 of 335" result is the best single piece of evidence for "behaviour, not knowledge".
2. **An honest negative result under a rule set in advance.** The candidate prints "no improvement" for their own best edit, reports the cost at options B, C and D, shows that two no-weight fixes are statistically indistinguishable from it (p ≥ 0.125), and lists the post-hoc parts in Limitations (note section 7, report section 7). This is what the brief means by "thoughtful negative results".
3. **Traceability and data-rights handling.** Checksums for weights, data and split. A frozen split that refuses to change. A claim-to-file map. Independent recomputation of about 120 numbers. A GPU re-score. Byte-identical report regeneration (`make reports`). A careful decision to withhold the copyrighted exam texts while shipping ids, answer letters and a rebuild path (`report/DATA_RIGHTS.md`, `NOTICE`). Another team could build on this directly.
