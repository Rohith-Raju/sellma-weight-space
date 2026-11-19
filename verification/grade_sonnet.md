# Independent grade (Sonnet)

## 1. Scores per criterion

**1. Scientific judgment: 5/5.** The note separates a behaviour fix from a knowledge gain with a direct test. Each question is asked four times with the right answer at A, B, C and D. The key evidence is "Gemma got 109 questions wrong in all four orders... after the weight edit... this group grew to 115". The note admits the edit's A-position gain is "paid for" at B/C/D (net +1 point) and that averaging over orders does as well. Section 7 names the missing positive control, single runs and selection on unseen papers. Minor: the loose claim "Its gap in Swedish looks like missing knowledge" rests on one model and one exam format.

**2. Experimental design: 4/5.** The pre-registered rule (`config/decision.toml`), a dev/test split by training cutoff, random same-size controls, a prompt-note baseline, order averaging, a group-LoRA baseline and a 4-bit check cover almost everything the brief lists. Weaknesses: the chosen edit was picked among four trials using test-paper results, and the four-order test was added post hoc (both disclosed). Test sets are small (120 and 320 questions). The AI editors used only 5, 7 and 11 of their runs and submitted nothing, which weakens the "article's method" comparison. Too many arms (Qwen, single-question edits, group edit, editors, Belebele, MMLU, perplexity) dilute a 4-hour question.

**3. Technical execution: 4/5.** The pipeline is thorough: frozen split with a checksum, hashes of model files, a second GPU rescoring that matched to 1e-8, and unit tests. The nine verifiers found a few errors (123 vs 124, 21/17 vs 18/14, 0.015% vs 0.0085%) and these were corrected. Deductions: the loop-rate damage check was too noisy to use. Full-precision and 4-bit perplexity are not comparable. The headline edit is a rank-1 change in a single block that the AI editor never submitted. The whole thing is far beyond a 4-hour scope, and the "about 4 hours of my own time" claim is hard to believe given the volume.

**4. Independent thinking: 5/5.** The candidate did not reproduce the paper. They tested the article's own caveat ("could remove an obstacle... could not add an ability") on a Swedish target. They found the option-A bias, which none of the planned tests would have found, and reframed the study around it. The recommendation (distillation, strongest multilingual base, a concrete next experiment with a positive control) is specific and honest that nothing was tested for training.

**5. Communication and reproducibility: 3/5.** The note is clear in plain language and has a good figure. But it is about 1,800 words, roughly twice the "approximately 1-2 pages" asked for. Many files (README 2.5k words, REPRODUCE 3k, CONFIG 2k, results README 1.8k, a 4.9k-word report) bury the reader. The exam question texts are not shipped, so reproduction needs `make data` against studera.nu. Most important, the repo state looks unpublished (see weaknesses).

## 2. Overall: 4.5/5 (round to 4 if the repo is not public and committed)

Verdict: yes, the candidate would advance, probably to the final stage. This is a strong, honest negative result with a real insight (the edit shifts answer-position behaviour, not Swedish knowledge) and unusually careful controls and verification. Against it: the note is twice the requested length, the whole study overran a 4-hour scope, the central edit was chosen post hoc, and the submission as it sits in git is not a public, committed repo. A grader who values "small, insightful" over "comprehensive" will mark this down for sprawl, not for science.

## 3. Five biggest weaknesses and fixes (by impact)

1. **The repo is not submittable as it stands.** `git status` shows only 2 commits, no remote, and most of the work (src, results, report, note) untracked. `config/decision.toml` is also modified after the commit that the note cites as "committed... about six hours before the first run". Fix: commit everything, push to a public GitHub repo, and add the URL to the top of the note. Either commit the 4-line `decision.toml` diff separately and say so, or reword the pre-registration claim to the exact commit hash `232bdc9` plus "plus a later 4-line edit (hash X)". Check the claim against the committed file before submitting.
2. **The note is too long.** About 1,830 words with eight sections plus a nine-step experiment list. Fix: cut to about 900-1,000 words. Replace the step list with a 4-row table (arm / what / result). Drop the Belebele/MMLU/perplexity details to one sentence each with "(report s.4)". Move the three-part recommendation to two sentences plus the next experiment. Put the single key number in the first paragraph: "109 always-wrong questions before, 115 after."
3. **Scope and the "4 hours" claim.** The work includes 2 models, 5 edit types, 9 verifiers and 8 report sections. Fix: add to section 8 one line splitting the time: "Of the ~4 h of my time, X h was design, Y h review. The runs (Z GPU-hours, mostly unattended) and the agents' work were on top." Also state in section 7 which parts a 4-hour version would keep (the four-order test and the controls only).
4. **Post-hoc selection weakens the headline edit.** The chosen edit was picked among four trial edits using test results, and the editor never submitted it. Fix: in section 4 put one sentence up front: "The edit was selected on test papers, so its test numbers are optimistic. The clean evidence is the 8 held-out older papers (27 fixed, 0 broken at A)." Also say plainly that the editors' budget use (5/7/11 of 10/25/50) means the article's search method was barely exercised, so the note says nothing about weight-space programming at scale.
5. **Reproducibility needs a short path.** The brief asks for "a straightforward way to reproduce your main experiment". REPRODUCE.md has 10 steps, a GPU, and a re-download of copyrighted PDFs. Fix: put a 5-line "Quick start" at the top of README: `uv sync; make test; make reports; make figures` (no GPU or data) and a one-line pointer to `make main`. State the expected runtime and GPU memory for `make main`. Reduce README, REPRODUCE, CONFIG and results/README overlap.

Smaller points: the note's "Opus 5.5" naming and "Borggren et al., EMNLP 2026" style citations should be checked to resolve; the sentence "It matters because a weight edit takes seconds, while training... takes thousands of GPU hours" is asserted without a source in the note; "Hypothesis" does not say what result would challenge it in one line (it does, in the long list of conditions, but not as a clear falsifier statement).

## 4. Requirement checklist

| Item | Status |
|---|---|
| Public GitHub repo | **Missing/at risk**: no remote, work untracked |
| Code and a straightforward reproduction | Partly: complete and tested, but multi-step, needs GPU and `make data` |
| Results: examples, outcomes | Met (`results/`, per-question answers, `results/README.md` mapping claims to files) |
| Results: config, model revisions, seeds | Met (`CONFIG.md`, revisions and shard hashes, seeds table) |
| Research note, ~1-2 pages | Partly: present and clear, but ~1,830 words |
| PDF report from the note | Met (`report/report.pdf`, 8 pages, built from `report_full.md`) |
| AI usage description | Met (note s.8 and report s.8), but time breakdown missing |
| Data and weights rights | Met: question texts withheld, `DATA_RIGHTS.md`, `NOTICE`, Gemma terms for delta files. Slightly cautious: reproduction depends on re-download |

## 5. Three strongest parts

1. The four-order (position-rotation) test and the always-wrong-group result (109 before, 115 after). It answers the brief's own question, "new capability or removal of a failure?", with direct evidence on the candidate's own setup.
2. The pre-set decision rule, the date-based split and the matched random controls, plus a nine-agent verification pass whose corrections are listed openly in `verification/SUMMARY.md`.
3. The honest framing: it reports a negative result, names the exploratory and post-hoc steps, flags the missing positive control, and gives a concrete next experiment.
