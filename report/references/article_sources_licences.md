# Article, SE.LLMA, data sources and licences: verified references (collected 2026-11-19)

I checked every reference below by opening it on 2026-11-19, using WebFetch, curl, or text pulled from the PDFs studera.nu serves. Two things need action before you submit. `exams/questions.jsonl`, which holds all the question texts, is not covered by `.gitignore`. And two of our planned claims are wrong as worded: the "only reproduction is secondary" line, and "the test exams were written after both models' training ended".

## Verified references

**[takehi2026]** Takehi, R., Teoh, J., Zhang, A., Khattab, O. (MIT CSAIL) (2026, 2 Nov). *Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights.* Personal blog post, not peer reviewed; a "Response to Comments" section was added 12 Nov 2026. https://rikiyat.github.io/blog/weight-space-programming/ (accessed 2026-11-19)
- **Exp 8 setup:** "Model: Qwen3.5-0.8B, thinking on (24 layers, 0.8 billion weights)". The edit: "Astra traced the looping to one direction written by the MLPs of layers 15 and 19 … and subtracted that direction from 64 output columns in each (131,072 weights)". Also: "the edit changes only 131,072 of the model's 0.8 billion weights (0.016%)."
- **Exp 8 table:**

| Run | MATH-500 | GSM8K | IFEval | Hit token limit |
|---|---|---|---|---|
| Base | 7.9 | 13.8 | 21.0 | 93.0 |
| 10 passes | 12.3 | 25.7 | 49.0 | 87.2 |
| 25 passes | 14.1 | 25.1 | 39.9 | 86.4 |
| 50 passes | 48.1 | 67.1 | 62.3 | 33.1 |

  Also: "with 10 or 25 forward passes the loops barely went away". One forward pass is defined as: "the model answers up to 512 problems, or reads up to 1,024 texts".
- **Gemma 3 270M:** This section is headed "Additional experiment" and is not numbered "8b", so don't call it that. MATH-500 went from base 6.8 to 7.0 / 7.6 / 7.7 / 7.0 at 5 / 10 / 25 / 50 passes, so the best gain is +0.9. The post says "none moved MATH-500 by more than about a point, a gain Astra itself reported as not established."
- **Exact wording of the key sentence:** "Editing weights could remove an obstacle that hid an ability, but it could not add an ability the model lacked."
- **Sycophancy (Exp 7):** "from the weights alone, Astra was unable to tell. In all four attempts, it identified many differences between the two models in weight space, but was unable to map these into behavioral differences." The authors then allowed attention patterns "to relax the experiment", and Astra picked out the sycophant "in all three sets of pairs".
- **Replication:** The post says "many people have replicated it … These are only a few of the more than a hundred we know of." This count is unverified. Logs are at github.com/RikiyaT/lm_understand-logs (I did not open it).
- **Supports:** all the Exp 8, Gemma, sycophancy and venue claims.

**[rikiyat_hf]** RikiyaT (2026). *Qwen3.5-0.8B-weight-space-programming* (model card). Hugging Face. https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming (accessed 2026-11-19)
- The changed tensors are `model.language_model.layers.{15,19}.mlp.down_proj.weight`. So "MLP layers 15 and 19" means precisely `down_proj`, 64 output columns each.
- Licence is Apache-2.0.
- The card also reports an independent re-run on full MATH-500: edited model 51.0%, with 31.6% of answers hitting the token limit.

**[coletti_hf]** JonathanColetti (2026). *Qwen3.5-0.8B-Astra-WeightEdit* (model card with benchmark). Hugging Face. https://huggingface.co/JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit (accessed 2026-11-19)
- This is the primary source for the independent reproduction. It says "Not affiliated with or endorsed by the authors, OpenAI or Qwen", and "I only applied their released diff and checked that it matches their evaluated checkpoint hash for hash."
- **Thinking on, base → edited:** MATH-500 7.4 → 50.2, GSM8K 13.0 → 67.6, IFEval 19.2 → 62.1.
- **Thinking off, base → edited:** MATH-500 45.8 → 43.6, GSM8K 57.1 → 55.7, IFEval 52.9 → 51.2.
- **Paired difference with thinking off (95% CI):** MATH-500 −2.2 [−4.3, −0.1], GSM8K −1.4 [−3.4, +0.6], IFEval −1.7 [−4.2, +0.9].
- **Edited with thinking on vs base with thinking off:** MATH-500 +4.3 [1.9, 6.8].
- **Settings:** thinking on used 32k tokens at temperature 0.6; thinking off used 8k tokens, temperature 0.7, top-p 0.8.
- The card also says "This removes a failure mode; it doesn't add capability." It lists the edit as CC BY 4.0 and the base model as Apache 2.0.
- **Supports:** critique section (e) and the loop-removal reading.

**[magica2026]** Editorial Team (2026, 3 Nov). *Qwen Weight Edit Reproduces Big Gains, but Settings Narrow the Gap.* Magica (commercial news site). https://magica.com/news/independent-tests-limit-mit-gpt-6-weight-editing-gains (accessed 2026-11-19)
- The table matches [coletti_hf].
- It contains the misstatement we planned to point out: "The edited model still beats the original with thinking disabled, but by 4–11 percentage points across three tests." Its own body text says the edit "reduced scores across all three tests" when thinking is disabled in both.
- **Supports:** the planned point that the article misstates the table.

**[hu2026]** Hu, J., Tang, C., Li, W., Zhu, H., Feng, C.-M., Zhang, Q., Chen, Y., Hu, M. (2026). *Exploring Weight-Space Programming: Capabilities, Generalization, and Limitations.* arXiv:2611.09382v1 (preprint, submitted 6 Nov 2026). Authors are from Shanghai AI Laboratory, Tongji University and Fudan University. https://arxiv.org/abs/2611.09382 (accessed 2026-11-19)
- The abstract says gains reach "up to 29.45, 4.42, 36.72, and 17.11 percentage points" on MATH-500, GPQA Diamond, IFEval and SciBench, and that "modifying only 1,023 parameters yields a 29.45-point improvement."
- Held-out MATH-500 went from 8.35 to 37.80, from one column of the layer-23 MLP down-projection.
- Each search "allows up to 50 forward-tool requests". No weights-only condition is reported.
- Reruns 1 and 2 gave "+5.00 and +27.30 points".
- It does not reproduce the blog's layer 15/19 edit.
- **Supports:** the claim that Hu et al. is entirely on the feedback side.

**[sellma]** WASP (n.d.). *SE.LLMA – Swedish large language models* (project page). https://wasp-sweden.org/research/projects/sellma/ (accessed 2026-11-19)
- Goal: "Develop and provide language models with strong Swedish language capabilities by using high-quality training data from the public sector, Swedish authors, publishers, journalists, and news media companies."
- Also: "The project will run for two years." It is led by researchers from "Linköping University, Umeå University, and Uppsala University, along with researchers from NAISS…"
- Supporting pages:
  - [wasp_news] WASP (2026, 23 Feb). *New initiative aims to give Sweden its own large-scale AI language model.* https://wasp-sweden.org/new-initiative-aims-to-give-sweden-its-own-large-scale-ai-language-model/ (funding up to SEK 30M from WASP)
  - WASP (2026, 2 Jun). *Learn more about SE.LLMA.* https://wasp-sweden.org/learn-more-about-se-llma/ ("…by training Swedish language models")

**[uhr_fpn]** Universitets- och högskolerådet (UHR). *Öva på tidigare högskoleprov* and the FPN archive. studera.nu. https://www.studera.nu/hogskoleprov/om/forbereda/tidigare/ and https://www.studera.nu/hogskoleprov/fpn/ (accessed 2026-11-19)
- "Det är gratis att ladda ner tidigare provhäften och facit."
- studera.nu: "Det är myndigheten Universitets- och högskolerådet (UHR) som ansvarar för den" (https://www.studera.nu/om-studera.nu/om-studera.nu/).
- **Supports:** where the exams and answer keys come from.

**[uhr_booklet]** UHR (2026). *Högskoleprovet, Provpass 2, Verbal del* (hp-2026-10-18). https://www.studera.nu/globalassets/05-hogskoleprovet/hp-2026-10-18/provpass-2-verb-utan-elf.pdf (accessed 2026-11-19). The same sentence is on p. 1 of hp-2025-10-19/provpass-3-verb-utan-elf.pdf.
- Exact text, p. 1: "Tillstånd har inhämtats att publicera det upphovsrättsligt skyddade material som ingår i detta prov."

**[uhr_ht2025]** UHR. *Provfrågor, facit och normering hösten 2025.* https://www.studera.nu/hogskoleprov/fpn/provfragor-facit-och-normering-hosten-2025/ (accessed 2026-11-19)
- "Universitets- och högskolerådet (UHR) har bara tillstånd att visa de engelska texterna (ELF) i en vecka efter provdagen." This explains the "utan ELF" file names.
- The linked source list (https://www.studera.nu/globalassets/05-hogskoleprovet/hp-2025-10-19/kallhanv.pdf) shows that the LÄS and MEK items come from third-party works. Examples: Birgitta Lindqvist, *Kinesisk ask* (Norstedts, 1997); Johannes Edfelt, poems (Bonniers, 2004); Svenska Dagbladet 2016-03-14.

**[sfs1960:729]** *Lag (1960:729) om upphovsrätt till litterära och konstnärliga verk*, consolidated through SFS 2026:1120. riksdagen.se. https://www.riksdagen.se/sv/dokument-och-lagar/dokument/svensk-forfattningssamling/lag-1960729-om-upphovsratt-till-litterara-och_sfs-1960-729/ (accessed 2026-11-19)
- **22 §:** "Var och en får citera ur offentliggjorda verk i överensstämmelse med god sed och i den omfattning som motiveras av ändamålet." This covers quoting a single example item.
- **15 a §:** "Den som har lovlig tillgång till ett verk får framställa exemplar av verket för text- och datautvinningsändamål. Exemplaren får inte behållas längre än vad som är nödvändigt…" **15 b §** gives research organisations the same right for research purposes.
- **26 §:** "Var och en får återge vad som muntligen eller skriftligen anförs … 2. i statliga eller kommunala representationer". The same section adds "en upphovsman har ensamrätt att ge ut samlingar av sina anföranden". This covers the Riksdag speeches.

**[riksdagen_terms]** Sveriges riksdag (2023, 18 Sep). *Användningsvillkor* (Riksdagens öppna data). https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anvandarstod/anvandningsvillkor/ (accessed 2026-11-19)
- "Riksdagens öppna data är fria att användas och spridas vidare så länge du anger källa."
- Attribution must read "Källa: Sveriges riksdag" or "Source: The Swedish Parliament".
- Services must not suggest the Riksdag is behind them.
- "Riksdagens öppna data innehåller personuppgifter", so GDPR and lag 2018:218 apply.
- Material protected under upphovsrättslagen "får inte återges" without permission.

**[riksdagen_anf]** Sveriges riksdag. *Anföranden 2025/26*, JSON bulk file. https://data.riksdagen.se/dataset/anforande/anforande-202526.json.zip (accessed 2026-11-19)
- The URL returns HTTP 200 and a 23,910,749-byte file. The listing page (https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/anforanden/) shows "json zip (22.8 MB)", last updated 2026-11-03.
- Because the file is still being updated, our sha256 will not match a later download.

**[gemma3_card]** Google DeepMind / Gemma Team. *Gemma 3 model card.* https://gemma.ai/models/gemma-3/ (accessed 2026-11-19; the old ai.google.dev/gemma/docs/core/model_card_3 now redirects here)
- "4B model was trained with 4 trillion tokens"
- "The knowledge cutoff date for the training data was August 2024."

**[gemma3_hf]** Google (2025). *google/gemma-3-4b-it*, revision 093f9f388b31de276ce2de164bdc2081324b9767. Hugging Face. https://huggingface.co/google/gemma-3-4b-it/tree/093f9f388b31de276ce2de164bdc2081324b9767 (accessed 2026-11-19)
- The revision exists: commit "Update README (#25)", lastModified 2025-03-21. Licence is "gemma", with manual gating.
- Model-card citation: Gemma Team, "Gemma 3", 2025.
- **Caveat:** the README at this revision does **not** contain the cutoff sentence. Cite [gemma3_card] for August 2024.

**[gemma_terms]** Google (last modified 1 Apr 2026). *Gemma Terms of Use.* https://gemma.ai/resources/terms/ (accessed 2026-11-19)
- §2.2: "You may use, reproduce, modify, Distribute…"
- If you distribute Gemma or a Model Derivative (§3.1), you must:
  - include the §3.2 use restrictions in your terms;
  - give recipients a copy of the Agreement;
  - mark modified files ("prominent notices stating that you modified the files");
  - add the notice "Gemma is provided under and subject to the Gemma Terms of Use found at gemma.ai/resources/terms".
- The Prohibited Use Policy is at gemma.ai/resources/prohibited_use_policy.

**[qwen35_hf]** Qwen Team (2026). *Qwen/Qwen3.5-9B*, revision c202236235762e1c871ad0ccb60c8ee5ba337b9a. Hugging Face. https://huggingface.co/Qwen/Qwen3.5-9B (accessed 2026-11-19)
- Licence is apache-2.0. The LICENSE file reads "Apache License Version 2.0, January 2004" and "Copyright 2026 Alibaba Cloud".
- The revision is the commit "Update README.md" by littlebird13, dated 2 Mar, and is the current main.
- The API shows createdAt 2026-02-27 and lastModified 2026-03-02.
- Model-card citation: Qwen Team, "Qwen3.5: Towards Native Multimodal Agents", February 2026, https://qwen.ai/blog?id=qwen3.5. That blog page did not render for me.
- **The card has no knowledge-cutoff statement.**

**[qwen35_gh]** QwenLM (2026). *Qwen3.5* repository README, News section. GitHub. https://github.com/QwenLM/Qwen3.5 (accessed 2026-11-19)
- "2026-03-02: Qwen3.5-9B, Qwen3.5-4B, Qwen3.5-2B, and Qwen3.5-0.8B are now available on Hugging Face Hub and ModelScope!"
- "2026-02-16: We release Qwen3.5."
- No cutoff is stated here either.

**[belebele]** Bandarkar, L., Liang, D., Muller, B., Artetxe, M., Shukla, S. N., Husa, D., Goyal, N., Krishnan, A., Zettlemoyer, L., Khabsa, M. (2024). *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.* ACL 2024, pp. 749–775. Dataset: https://huggingface.co/datasets/facebook/belebele/tree/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2 (accessed 2026-11-19)
- The revision exists: commit "Update README.md with ACL citation (#8)".
- `license: cc-by-sa-4.0`. The `swe_Latn` config is `data/swe_Latn.jsonl`, with "900 questions per language variant".
- The assembled training set "is governed by a different license". We don't use it.

**[mmlu]** Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., Steinhardt, J. (2021). *Measuring Massive Multitask Language Understanding.* ICLR 2021. Dataset: https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe (accessed 2026-11-19)
- The revision exists: commit "Upload dataset" by Long Phan. Licence is mit.
- The upstream LICENSE reads "MIT License, Copyright (c) 2020 Dan Hendrycks" (https://github.com/hendrycks/test/blob/master/LICENSE).

**[llamacpp]** ggml authors (2026). *llama.cpp*, release b8680. GitHub. https://github.com/ggml-org/llama.cpp/releases/tag/b8680 (accessed 2026-11-19)
- Published 2026-04-06T19:38:24Z at commit 15f786e6581598638840276948a7e6183fc96a83.
- LICENSE at that tag: "MIT License, Copyright (c) 2023-2026 The ggml authors".

## Planned claims the sources contradict or don't support

1. **"The only reproduction is secondary, reported in a commercial news article."** This is wrong. The reproduction is primary and checkable on JonathanColetti's model card [coletti_hf]. The blog's own model card [rikiyat_hf] also reports a re-run (51.0%), and the blog claims "more than a hundred" replications, which is unverified. Cite [coletti_hf] for the numbers and use [magica2026] only for its misstatement.
2. **"Each Qwen budget was run once."** The blog never says how many runs there were. It only says "varying the budget across 10, 25 and 50." Reword to "one result is reported per budget, with no repeat runs or variance."
3. **"Test exams were written after both models' training ended."** This can't be shown for Qwen. Qwen has published no cutoff, and the April 2025 date has no source. The 9B model came out on 2 March 2026, 10 months after that date, so a later cutoff is quite possible, and that could include the 2025-10-19 test exam. Call the April 2025 date an assumption, not a fact. For the release date, "2 March 2026" is accurate; the family launched 16 Feb 2026.
4. **Gemma cutoff source.** "August 2024" is correct, but it comes from the gemma.ai model card. The README at the pinned HF revision doesn't contain it.
5. **"Exp 8b".** That label is ours, not the blog's. The blog says "Additional experiment".
6. **Hu et al.** Confirmed: no weights-only condition, a different edit (layer 23, 1,023 values), and 37.80 rather than 48.1.

## Högskoleprovet: is not republishing the right call?

Yes. These are the grounds:
- The booklets reuse third-party copyrighted texts, listed in UHR's own source list. These include Norstedts and Bonniers literature, poems, and SvD articles.
- The booklet's permission sentence covers UHR's own publication only.
- UHR's permission for the English (ELF) texts is limited to one week, which shows how narrow these permissions are.
- I found no reuse terms anywhere on studera.nu.
- Swedish copyright law allows copies for text and data mining (15 a–b §), which covers our evaluation, and allows quoting (22 §). It does not allow redistributing the items.

So: link to studera.nu, report only aggregate scores, and quote at most a single short item as an example.

**Problem in our own repo:** `.gitignore` excludes `exams/raw/` but **not `exams/questions.jsonl`**, which contains the full question texts. If the repo is pushed, that counts as republishing. 15 a § also says copies may not be kept longer than the purpose requires.

## Riksdagen and the damage-check corpus

`data/damage/SOURCES.md` says no reuse terms were checked. They have now been verified [riksdagen_terms]:
- Use is free with the attribution "Källa: Sveriges riksdag".
- The data contains personal data, so GDPR applies.
- 26 § lets anyone reproduce speeches made in the Riksdag, but each speaker keeps the exclusive right to publish collections of their own speeches.

There is also an inconsistency in our files. README line 64 says the perplexity text is Wikipedia plus Riksdag speeches, but the SOURCES.md table shows all 75 documents coming from data.riksdagen.se.

## Licence obligations

- If we release an edited Gemma, the Gemma Terms §3.1 apply: pass on the use restrictions, include a copy of the terms, mark the modified files, and add the Notice file.
- Qwen, MMLU and llama.cpp are permissive (Apache-2.0 or MIT).
- Belebele is CC BY-SA 4.0, so give credit, and any adaptation must be shared under the same licence.

## Could not verify

- **Which SE.LLMA URL the assignment links to.** I couldn't find it. The three WASP pages above are the official sources, and the project page has the "strong Swedish language capabilities" wording.
- **Any studera.nu or UHR terms of use for reusing exam questions.** None exist on the site as far as I could find, so cite the booklet sentence and the source list instead.
- **The Qwen3.5 blog page** (qwen.ai) didn't render.
- **The blog's logs repo, and the Willison and VentureBeat coverage**, were not opened. Willison's piece is about Qwen 3.8 27B, not this edit.
## Article: all experiments, and what the agent could access (checked 2026-11-19 by re-opening the post)

| Exp | Task | Access | Result (verbatim where quoted) |
|---|---|---|---|
| 1 | Which game a network was trained on | weights as text only, no forward passes | "Astra was still able to name the right game for all four networks." |
| 2 | What a diffusion model generates | weights only, "We do not allow tool use or forward passes." | "Astra was able to tell what each model generates, from the weights alone." (drew outlines, not filled shapes) |
| 3 | Training language of a small transformer, head roles | 4 middle blocks only; Python but "may not run the network or any of its layers on any input" | "Astra names the right language for all three networks, in both of its attempts." |
| 4 | Fix a failing car-on-ramp policy | weights, scene description, one recorded episode; one reply, no tools | "Astra's edit makes the car succeed in all 1,024 episodes, changing 3.5% of the weights." |
| 5 | Write MinAtar game networks from scratch | rules, code, layout; never run before scoring | "Astra's generated networks beat PPO trained for 100 million steps in all three games." (mixed vs 500M) |
| 6 | Write a Go network | sandbox that cannot run the net; 0–50 forward passes (128 rollouts each); "keeps a change only if it wins more" | won >half its games with 0 passes; "With 50 forward passes, Astra's generated network beats the equal-sized PPO network." |
| 7 | Which LLM is sycophantic | weights + tokenizer + Python, no running; relaxed: attention patterns for ≤16 texts, ≤20 runs | "Unfortunately, from the weights alone, Astra was unable to tell." Relaxed: "able to pick out the sycophant in all three sets of pairs" |
| 8 | Qwen3.5-0.8B maths | Python over weights (no GPU), GSM8K/MATH practice prompts, 10/25/50 forward passes, "a harness like the one in Experiment 6"; "no training or gradient information" | big gain only at 50 passes; "with 10 or 25 forward passes the loops barely went away"; 10 passes still gave MATH 12.3, IFEval 49.0 |
| extra | Gemma 3 270M maths (model that does not loop) | 5/10/25/50 passes | "Here, Astra could not improve it." |

Exp 8 mechanism: "Astra traced the looping to one direction written by the MLPs of layers 15 and 19". The post does not say explicitly that Exp 8 kept only edits that helped; that wording is from Exp 6.
Thinking-off numbers (45.8 base, 43.6 edited) are from [coletti_hf], not the article. The article's comment section: "Turning thinking off raises the base score, but it does not fix the weights." "It avoids the loops by not using the mode that has them."
Coletti, edited thinking-on vs base thinking-off: MATH-500 +4.3 [1.9, 6.8], GSM8K +10.5 [8.3, 12.7], IFEval +9.2 [5.2, 13.2].
