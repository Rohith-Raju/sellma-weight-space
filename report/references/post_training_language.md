# Post-training and target-language ability: literature check

Compiled 2026-11-19 for the SE.LLMA research note. All sources below were opened on 2026-11-19 (arXiv abstract or HTML page, ACL Anthology page, or the PDF).

How quotes were collected. Quotes for Okapi (entry 6) and Holmström and Doostmohammadi (entry 7) were copied from text extracted directly from the PDF. Other quotes were pulled from the arXiv or Anthology page through a fetch tool that returns short excerpts. They are verbatim as returned, but each is short (under about 125 characters). Spot-check any quote against the paper before it goes into a final report.

Verdict labels:
- **IMPROVES** = supports "post-training improves target-language ability"
- **ELICITS** = supports "post-training mostly changes behaviour / elicits existing ability"
- **MIXED** = evidence goes both ways, or the gains are in behaviour but not in knowledge

Context for the reader. The SE.LLMA finding is that small targeted weight edits on Gemma 3 4B changed answer selection but added no knowledge on a Swedish reading and sentence-completion exam. The question here is whether SFT, DPO/RLHF, RL or distillation would do better, and on which kinds of ability.

---

## Verified references

### A. Multilingual and target-language post-training

#### 1. Shaham et al. 2024: a pinch of multilinguality
- **Citation:** Uri Shaham, Jonathan Herzig, Roee Aharoni, Idan Szpektor, Reut Tsarfaty, Matan Eyal. 2024. *Multilingual Instruction Tuning With Just a Pinch of Multilinguality.* Findings of ACL 2024 (venue taken from the arXiv comments field). https://arxiv.org/abs/2401.01854. Accessed 2026-11-19.
- **Setup:** PaLM 2-S tuned on 1,000 LIMA plus 3,640 OpenAssistant examples. Twelve languages including Finnish and Estonian. Evaluation is open-ended instruction following on 617 translated AlpacaFarm prompts, judged side by side by PaLM 2-L. Judge agreement with humans was 75 to 79.5% (paper, HTML v4).
- **Quotes:**
  - "only 40 multilingual examples integrated in an English tuning set substantially improve"
  - "multilingual instruction-following, both in seen and unseen languages during tuning."
  - "despite training on 10x fewer examples in those languages."
  - "diversifying the instruction tuning set with even just 2-4 languages significantly improves" ... "cross-lingual generalization."
- **Numbers:** 40 examples (about 1% of the set) give a significant boost. A uniform 12-language mix matches or beats monolingual tuning in 8 of 12 languages with 12x fewer examples per language. Exact win rates appear only in figures, so they are not given here.
- **What improved:** instruction following and response quality as judged by an LLM. The paper does not measure knowledge or exam accuracy.
- **Verdict: IMPROVES** (instruction following only). A very small amount of target-language data is enough.

#### 2. Kew, Schottmann and Sennrich 2024: how much multilinguality is needed
- **Citation:** Tannon Kew, Florian Schottmann, Rico Sennrich. 2024. *Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed?* Findings of EMNLP 2024 (arXiv comments). https://arxiv.org/abs/2312.12683. Accessed 2026-11-19.
- **Setup:** Mainly Llama 2 7B (also Llama 2 70B, Falcon 7B, Llama 3 8B). 3,200 English examples, with 200 per language swapped in for up to five languages. Languages include Icelandic.
- **Quotes:**
  - "multilingual instruction tuning is most beneficial for generative tasks that assume input/output language agreement."
  - On XNLI: "performance for all target languages remains uniform regardless of the amount of multilinguality used during finetuning."
  - On X-CSQA: "we do not observe any consistent gains for non-English target languages on the basis of multilingual instruction tuning."
  - The authors name "language exposure during pretraining being the main limitation to this generalisation ability."
- **Numbers:** Dialogue scores rise from one to two languages, then plateau at about three. Results are reported in figures, so no exact values are given here. Low-resource languages (Greek, Hindi, Icelandic) stay low whatever the mix.
- **Verdict: MIXED.** Post-training helps generation in the target language. It does not help classification or multiple-choice commonsense tasks. Pretraining exposure sets the ceiling. This is the closest match to an exam-style task.

#### 3. Üstün et al. 2024: Aya model
- **Citation:** Ahmet Üstün, Viraat Aryabumi, Zheng Yong, Wei-Yin Ko, Daniel D'souza, et al. (17 authors). 2024. *Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model.* ACL 2024 (Long Papers), pp. 15894–15939. https://aclanthology.org/2024.acl-long.845/ (arXiv 2402.07827). Accessed 2026-11-19.
- **Quotes:**
  - "follows instructions in 101 languages of which over 50% are considered as lower-resourced."
  - "Aya outperforms mT0 and BLOOMZ on the majority of tasks while covering double the number of languages."
  - On multilingual MMLU: "improving average accuracy by 21.1% over mT0x."
  - "The Aya model is preferred against mT0 and mT0x in all languages with an average of 87% and 86% win rates respectively."
- **Numbers (HTML v1):** On held-out XCOPA, XNLI, XStoryCloze and XWinograd, the average is 73.9 for Aya vs 65.8 for mT0x. Multilingual MMLU (5-shot) is 37.3 for Aya vs 30.8 for mT0x and 31.5 for mT0. FLORES X→En spBLEU is 29.1 vs 20.2. Human win rate is 77% over mT0 and mT0x.
- **Caveat:** Aya, mT0 and mT0x are all fine-tuned from the same mT5-13B base. The comparison is between fine-tuning mixtures, not between base and post-trained models. The MMLU gain (about 6.5 points) shows that the data mixture matters. It does not show that the knowledge is new to the base model.
- **Verdict: IMPROVES** (task following and generation). For knowledge it is MIXED, given the caveat above.

#### 4. Dang et al. 2024: Aya Expanse (preprint)
- **Citation:** John Dang, Shivalika Singh, Daniel D'souza, Arash Ahmadian, Alejandro Salamanca, et al. (45 authors). 2024. *Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier.* arXiv preprint 2412.04261 (not peer reviewed). https://arxiv.org/abs/2412.04261. Accessed 2026-11-19.
- **Quotes:** win rates "up to a 76.6% win-rate" against open-weight models of similar size on translated Arena-Hard. The 32B model beats Llama 3.1 70B with a "54.0% win-rate."
- **What improved:** chat preference win rates judged by an LLM across 23 languages. Swedish is probably not one of the 23. This is from memory of the Aya 23 language list and was not checked.
- **Verdict: IMPROVES** (preference and behaviour metrics). There is no knowledge measurement in the abstract. **Preprint.**

#### 5. Dang et al. 2024: RLHF Can Speak Many Languages
- **Citation:** John Dang, Arash Ahmadian, Kelly Marchisio, Julia Kreutzer, Ahmet Üstün, Sara Hooker. 2024. *RLHF Can Speak Many Languages: Unlocking Multilingual Preference Optimization for LLMs.* EMNLP 2024, pp. 13134–13156. https://aclanthology.org/2024.emnlp-main.729/ (arXiv 2407.02552). Accessed 2026-11-19.
- **Quotes:**
  - "Our preference-trained model achieves a 54.4% win-rate against Aya 23 8B,"
  - "a 69.5% win-rate or higher against widely used models like Gemma-1.1-7B-it,"
  - "we expand the frontier of alignment techniques to 23 languages covering half of the world's population."
- **What improved:** open-ended generation quality, measured as LLM-judged win rates.
- **Verdict: IMPROVES** (preference and behaviour). Knowledge is not measured.

#### 6. Lai et al. 2023: Okapi
- **Citation:** Viet Lai, Chien Nguyen, Nghia Ngo, Thuat Nguyen, Franck Dernoncourt, Ryan Rossi, Thien Nguyen. 2023. *Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback.* EMNLP 2023 System Demonstrations, pp. 318–327. https://aclanthology.org/2023.emnlp-demo.28/ (arXiv 2307.16039). Accessed 2026-11-19.
- **Quotes (from PDF text):**
  - "RLHF is generally better than SFT for multilingual fine-tuning of LLMs over different tasks, base models, and language groups. The improvement of average performance over all languages can go up to 2.5% on the HellaSwag dataset with LLaMA"
  - "the generated instructions cannot well activate/complement the language and knowledge skills related to MMLU from the LLMs to attain meaningful improvement from instruction tuning."
  - "The performance improvement of RLHF for low-resource languages is also the least"
- **Numbers (translated benchmarks, averages over languages, base → SFT → RLHF):**
  - LLaMA 7B, ARC: 33.3 → 35.2 → 37.3. HellaSwag: 46.4 → 47.1 → 49.6. **MMLU: 29.8 → 30.1 → 30.8.**
  - BLOOM 7B, ARC: 28.2 → 28.4 → 30.0. HellaSwag: 36.8 → 37.7 → 39.5. **MMLU: 27.1 → 26.6 → 26.9** (no gain, near the 25% chance level).
  - **Danish**, LLaMA 7B, ARC: 32.7 → 35.1 → 36.8. HellaSwag: 46.7 → 47.7 → 51.7. MMLU: 30.0 → 30.9 → 31.8.
- **Verdict: MIXED.** There are gains of 2 to 4 points on multiple-choice ARC and HellaSwag, which are simple-knowledge and commonsense tasks. There is about 1 point or less on MMLU, which is professional knowledge. The authors themselves say the instructions cannot add MMLU-type knowledge. This is direct evidence that post-training moves exam-style scores only a little.

### B. Swedish and Nordic

#### 7. Holmström and Doostmohammadi 2023: Swedish instruction fine-tuning
- **Citation:** Oskar Holmström, Ehsan Doostmohammadi. 2023. *Making Instruction Finetuning Accessible to Non-English Languages: A Case Study on Swedish Models.* NoDaLiDa 2023, pp. 634–642. https://aclanthology.org/2023.nodalida-1.62/. Accessed 2026-11-19.
- **Setup:** GPT-SW3 1.3B and OPT 1.3B fine-tuned on Unnatural Instructions machine-translated (DeepL) into Swedish.
- **Quotes (abstract and PDF):**
  - "the use of translated instructions significantly improves the models' zero-shot performance."
  - "we see that perplexity increases after instruction finetuning. It is especially pronounced for the smaller models"
  - "a strong foundation in the target language is helpful."
- **Numbers (Table 2, ROUGE-L):**
  - GPT-SW3: 0.084 → 0.542 on Unnatural Instructions and 0.009 → 0.124 on Natural Instructions.
  - OPT: 0.071 → 0.449 and 0.006 → 0.101.
  - Perplexity on new SVT news text (Table 1) got worse: GPT-SW3 1.92 → 2.66 and OPT 2.79 → 5.45.
- **Verdict: MIXED.** Following tasks in Swedish improved a lot. Swedish language modelling, as measured by perplexity, got worse after SFT on translated data.

#### 8. Zosa, Komulainen and Pyysalo 2025: post-training a Finnish LLM
- **Citation:** Elaine Zosa, Ville Komulainen, Sampo Pyysalo. 2025. *Got Compute, but No Data: Lessons From Post-training a Finnish LLM.* NoDaLiDa/Baltic-HLT 2025. https://arxiv.org/abs/2503.09407. Accessed 2026-11-19.
- **Setup:** Poro 34B base. SFT on OASST2 translated into Finnish, then DPO. Evaluated with IFEval only (510 prompts, English and DeepL-translated Finnish).
- **Quotes:**
  - "with a few hundred Finnish instruction samples we can obtain competitive performance in Finnish instruction-following."
  - DPO "did not improve the Finnish response rates compared to the SFT model."
- **Numbers (Finnish IFEval, strict instruction-level):**
  - SFT with 0% Finnish data: 31.41. With 10% Finnish (about 400 samples): 32.69. With 100% Finnish: 34.48.
  - DPO on both languages: 37.80.
  - The Finnish response rate went from 47.45% to 90.00% with 10% Finnish data.
- **Verdict: IMPROVES** (instruction following and answering in the right language). Knowledge is not measured.

#### 9. Samuel et al. 2026: on-policy alignment for Norwegian
- **Citation:** David Samuel, Lilja Øvrelid, Erik Velldal, Andrey Kutuzov. 2026. *Fluent Alignment with Disfluent Judges: Post-training for Lower-resource Languages.* ICLR 2026 (arXiv journal-ref). https://arxiv.org/abs/2512.08777. Accessed 2026-11-19.
- **Quotes:**
  - "The results show that the on-policy aspect is crucial and outperforms the alternatives"
  - "even a small amount of translated data can introduce a measurable amount of disfluencies in the final policy."
- **Numbers (Norwegian Bokmål, five native-speaker judges, 300 pairs):**
  - On-policy RL beat SFT on translated data on fluency 67.5% to 32.5%.
  - On-policy RL beat Mistral Nemo 12B Instruct 91.8% to 8.2%.
  - The automatic fluency score was 92.2% for on-policy RL, 85.7% for translated SFT and 65.3% for Mistral Nemo.
- **Verdict: IMPROVES** (fluency). Knowledge is not measured. This is the strongest Nordic evidence that the *method* of post-training matters (on-policy RL beats SFT on translated data) for target-language quality.

#### 10. Zhang et al. 2025: SnakModel (Danish)
- **Citation:** Mike Zhang, Max Müller-Eberstein, Elisa Bassignana, Rob van der Goot. 2025. *SnakModel: Lessons Learned from Training an Open Danish Large Language Model.* NoDaLiDa 2025 (oral, per arXiv comments). https://arxiv.org/abs/2412.12956. Accessed 2026-11-19.
- **Setup:** Llama 2 7B. Continued pretraining on 13.6B Danish words, then SFT on 3.7M Danish instructions, all machine-translated (Aya Collection, OpenHermes, OpenOrca subsets).
- **Quotes (HTML v1):**
  - After continued pretraining: "Meanwhile, we see performance decreases for NER and QA."
  - After instruction tuning: "instruction tuning recovers these original functionalities."
  - Gains are "particularly pronounced for sub-tasks based on natural Danish data."
- **Numbers (Danish ScandEval, Table 3. Columns: LA, NER, Senti, Summ, CSR, QA, TM = proverbs, CT = citizenship tests, then AVG):**
  - Llama 2 7B base: 33.43, 22.31, 61.54, 65.50, 29.76, 63.54, 38.69, 57.05. AVG 46.48.
  - Llama 2 7B chat: 47.42, 24.63, 62.35, 66.15, 32.24, 61.34, 46.67, 55.18. AVG 49.50.
  - SnakModel base (continued pretraining only): 56.28, 19.91, 57.42, 58.95, 30.47, 18.52, 69.14, 60.93. AVG 46.45.
  - SnakModel instruct: 52.91, 29.76, 66.70, 66.61, 29.46, 64.66, 71.05, 71.88. AVG 56.63.
- **Reading the ablation:**
  - Continued pretraining did most of the work on culture-specific knowledge: proverbs (TM) went from 38.69 to 69.14, and acceptability (LA) from 33.43 to 56.28.
  - Instruction tuning then added +10.95 on citizenship tests (CT) and +46 on QA. The QA jump is mostly a recovery of answer format after continued pretraining broke it.
  - Commonsense (CSR, translated HellaSwag) did not move at any stage (29.76 → 30.47 → 29.46). LA fell 3.37 points after instruction tuning.
- **Verdict: MIXED.** Instruction tuning raised the ScandEval average by 10.18 points over the continued-pretraining checkpoint. Much of that gain is format recovery and task following. The big knowledge gain (proverbs) came from continued pretraining.

#### 11. Mikhailov et al. 2025: NorEval, base vs instruct
- **Citation:** Vladislav Mikhailov, Tita Enstad, David Samuel, Hans Christian Farsethås, Andrey Kutuzov, Erik Velldal, Lilja Øvrelid. 2025. *NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark.* Findings of ACL 2025, pp. 3495–3541. https://aclanthology.org/2025.findings-acl.181 (arXiv 2504.07749). Accessed 2026-11-19.
- **Quotes:**
  - "instruction-tuning can yield both positive and negative effects depending on the task."
  - "At the same time, it can degrade the performance on tasks requiring Norwegian language knowledge"
- **Numbers (Table 4, normalised scores, base → instruct):**
  - AI-Sweden/Llama-3-8B: overall 51.3 → 45.7. Norwegian language knowledge 51.0 → 16.1. World knowledge 54.8 → 55.3. Reading comprehension 47.1 → 53.9. Instruction following 38.1 → 50.1.
  - NorMistral-7B-warm: language knowledge 59.2 → 16.9. Instruction following 14.6 → 56.1. Machine translation 72.0 → 30.5.
  - Meta Llama-3-8B: overall 47.0 → 48.2. Mistral-Nemo-12B: overall 47.6 → 52.1.
  - In all five pairs, instruction following rose and Norwegian language knowledge and translation fell.
- **Verdict: MIXED / ELICITS.** This is the clearest Nordic base-vs-instruct comparison. Instruct versions follow instructions better. They lose a lot on language-knowledge tasks (probably partly a prompt-format effect, which the table cannot separate). World-knowledge scores barely change. Note that one pair is AI Sweden's Llama-3-8B.

### C. Does post-training add knowledge or elicit it?

#### 12. Gao et al. 2024: cross-lingual knowledge alignment is shallow
- **Citation:** Changjiang Gao, Hongda Hu, Peng Hu, Jiajun Chen, Jixing Li, Shujian Huang. 2024. *Multilingual Pretraining and Instruction Tuning Improve Cross-Lingual Knowledge Alignment, But Only Shallowly.* NAACL 2024 (Long), pp. 6101–6117. https://aclanthology.org/2024.naacl-long.339 (arXiv 2404.04659). Accessed 2026-11-19.
- **Quotes:**
  - "neither multilingual pretraining nor instruction tuning can substantially improve the cross-lingual knowledge conductivity."
  - "Surprisingly, both English-only and multilingual instruction tuning causes drop in the RA scores."
  - "compared with English-only tuning, multilingual tuning causes less damage to the factual knowledge performance."
- **Numbers (factual-knowledge score change vs the LLaMA foundation model, Table 10):**
  - Vicuna v1.5: English −7.94, Chinese −11.11, other languages −2.30.
  - LLaMA2-Chat: English −10.90, Chinese −15.13.
  - Basic language ability rose with multilingual tuning (Vicuna v1.5: +8.24 for Chinese, +10.17 for other languages).
- **Verdict: ELICITS.** Instruction tuning raised basic target-language ability. It lowered factual-knowledge scores and did not improve knowledge transfer across languages.

#### 13. Chua et al. 2025: crosslingual knowledge barriers
- **Citation:** Lynn Chua, Badih Ghazi, Yangsibo Huang, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Amer Sinha, Chulin Xie, Chiyuan Zhang. 2025. *Crosslingual Capabilities and Knowledge Barriers in Multilingual Large Language Models.* COLM 2025 (venue per the Google Research publication page). https://arxiv.org/abs/2406.16135. Accessed 2026-11-19.
- **Quotes:**
  - "fine-tuning of LLMs on mixed-language data, which effectively reduces these gaps"
  - "fine-tuning on English WikiText-103 corpora hurts the performance"
- **Numbers (MMLU with mixed-language questions, 0-shot):**
  - Llama3-8B dropped from 60.54 in English to 48.62 when mixed-language. GPT-4 dropped from 81.82 to 68.61.
  - Fine-tuning on mixed-language WikiText raised Llama3-8B from 48.62 to 51.75 and Llama2-7B from 32.18 to 34.08.
  - English-only fine-tuning lowered both (47.83 and 31.46).
- **Verdict: IMPROVES** (modestly). Fine-tuning on mixed-language text recovers 2 to 3 points of knowledge access in another language. It does not close a 12-point gap. Note that this is fine-tuning on raw text, which is closer to continued pretraining than to instruction tuning.

#### 14. Yue et al. 2025: does RL incentivise reasoning beyond the base model?
- **Citation:** Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, Gao Huang. 2025. *Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* NeurIPS 2025 (oral). https://arxiv.org/abs/2504.13837. Accessed 2026-11-19.
- **Quotes:**
  - "Surprisingly, we find that the current training setup does not elicit fundamentally new reasoning patterns."
  - "the observed reasoning abilities originate from and are bounded by the base model."
  - "RLVR increases the likelihood of sampling correct responses at low k, but narrows the model's overall coverage."
  - "distillation can introduce new reasoning patterns from the teacher"
- **Numbers (Qwen2.5-7B, Table 3):**
  - MATH500 pass@1 went from 34.5 (base) to 74.4 (GRPO). Pass@256 went from 96.2 to 97.2.
  - Omni-MATH-Train pass@256 went from 67.2 (base) to 66.3 (GRPO).
  - On AIME24 at k=1024, 13.3% of problems were solved only by the base model and 0.0% only by the RL model.
- **Verdict: ELICITS** (for RL with verifiable rewards). The same paper points to distillation as the route that does add new ability.

#### 15. DeepSeek-AI 2025: DeepSeek-R1
- **Citation:** DeepSeek-AI (Daya Guo, Dejian Yang, Haowei Zhang, et al.). 2025. *DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning.* Nature 645, 633–638. DOI 10.1038/s41586-025-09422-z. Read through arXiv 2501.12948 (journal-ref confirms Nature). The Nature page redirected to a login and was not read. https://arxiv.org/abs/2501.12948. Accessed 2026-11-19.
- **Quotes (arXiv v1 HTML):**
  - AIME 2024 pass@1 for R1-Zero went "from an initial 15.6% to an impressive 71.0%"
  - "CoT often exhibits language mixing, particularly when RL prompts involve multiple languages."
  - On the language consistency reward: "ablation experiments show that such alignment results in a slight degradation in the model's performance"
  - "distilling more powerful models into smaller ones yields excellent results"
- **Numbers (Qwen-32B, Table 6, RL from scratch vs distilled from R1):**
  - AIME 2024 pass@1: 47.0 vs 72.6.
  - GPQA Diamond: 55.0 vs 62.1.
  - LiveCodeBench: 40.2 vs 57.2.
- **Verdict: MIXED.** Large RL gains on reasoning are possible for a very strong base. For a small model, distillation from a stronger teacher beat RL on every reported benchmark. Forcing the reasoning to stay in one language cost some accuracy.

#### 16. Park et al. 2025: cross-lingual collapse under RLVR (preprint)
- **Citation:** Cheonbok Park, Jeonghoon Kim, Joosung Lee, Sanghwan Bae, Jaegul Choo, Kang Min Yoo. 2025. *Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models.* arXiv preprint 2506.05850 (v3, Feb 2026, not peer reviewed). https://arxiv.org/abs/2506.05850. Accessed 2026-11-19.
- **Quotes:**
  - "under RLVR, CoT in LLMs systematically drifts toward the pre-training dominant language as reasoning performance rises"
  - interventions that favour target-language traces "reveal a persistent performance-fidelity trade-off"
- **Numbers (Llama-3.2-3B, Ukrainian GSM8K, Table 9):**
  - Plain GRPO reached 70.9% accuracy, but only 0.3% of reasoning words were Ukrainian.
  - Adding a language-proportion reward raised Ukrainian to 81.6% of words and cut accuracy to 62.5% (−8.4 points).
- **Verdict: ELICITS / caution.** RL on target-language prompts improves the task but pulls reasoning back into English. Keeping it in the target language costs accuracy. **Preprint.**

#### 17. Liu et al. 2025: ProRL (preprint, counter-evidence)
- **Citation:** Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, Yi Dong. 2025. *ProRL: Prolonged Reinforcement Learning Expands Reasoning Boundaries in Large Language Models.* arXiv preprint 2505.24864 (no venue listed). https://arxiv.org/abs/2505.24864. Accessed 2026-11-19.
- **Quotes:** prolonged RL "can uncover novel reasoning strategies that are inaccessible to base models." The abstract also says gains correlate with the base model's task competence and with training duration.
- **Numbers:** none in the abstract. Not extracted from the full text.
- **Verdict: IMPROVES** (counter-evidence to Yue et al., for reasoning in English on a 1.5B model). **Preprint.** It does not address language or knowledge.

#### 18. Raghavendra, Nath and Hendryx 2024: revisiting the superficial alignment hypothesis (preprint)
- **Citation:** Mohit Raghavendra, Vaskar Nath, Sean Hendryx. 2024. *Revisiting the Superficial Alignment Hypothesis.* arXiv preprint 2410.03717 (no venue listed). https://arxiv.org/abs/2410.03717. Accessed 2026-11-19.
- **Quotes:**
  - "a model's ability to integrate new knowledge greatly improves" with post-training, on tasks such as multihop QA.
  - The Superficial Alignment Hypothesis is "at best, an over-simplification."
- **Numbers:** The abstract reports power-law scaling of post-training performance with the number of fine-tuning examples (Llama-2, Llama-3, Mistral) but gives no values.
- **Verdict: IMPROVES** (counter to LIMA and URIAL). **Preprint.** It is about English and about reasoning and knowledge *use*, not language.

#### 19. Mecklenburg et al. 2024: injecting new knowledge via SFT (preprint)
- **Citation:** Nick Mecklenburg, Yiyou Lin, Xiaoxiao Li, Daniel Holstein, Leonardo Nunes, Sara Malvar, Bruno Silva, Ranveer Chandra, Vijay Aski, Pavan Kumar Reddy Yannam, Tolga Aktas, Todd Hendry. 2024. *Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning.* arXiv preprint 2404.00213. https://arxiv.org/abs/2404.00213. Accessed 2026-11-19.
- **Quotes (HTML v2):**
  - "In none of the document/dataset-scaling configurations do we outperform RAG with fine-tuning"
  - "Fact-based scaling, on the other hand, offers a more systematic approach to ensure even coverage across all facts."
- **Numbers:** GPT-4 (0613) fine-tuned with LoRA on 2023 sports events after its knowledge cutoff. Base accuracy on post-cutoff events was at most 0.242. Token-based scaling from 1x to 5x gave +78.6% relative on one event. SFT was "within 16% (or better) of RAG performance for all scenarios." Exact fact-based SFT accuracies appear only in figures.
- **Verdict: MIXED.** SFT can inject narrow new facts when the data is built fact by fact. It still did not beat retrieval. **Preprint.**

---

## What the evidence supports for SE.LLMA

- **Post-training reliably improves *how* a model uses a language, not *what* it knows in it.** Every target-language study here found clear gains in instruction following, response language, fluency or judged preference (Shaham 2024, Zosa 2025, Samuel 2026, Dang 2024, Holmström 2023). On knowledge-style multiple-choice tests the gains are small or absent. Okapi's MMLU moved +1.0 point with LLaMA and −0.2 with BLOOM. Kew 2024 found no consistent gains on XNLI or X-CSQA.
- **For a Swedish reading and sentence-completion exam, the main lever is probably target-language pretraining, not post-training.** In SnakModel, continued pretraining raised Danish proverbs from 38.69 to 69.14. Instruction tuning then mainly restored answer format and added about 11 points on citizenship tests. Commonsense (HellaSwag) did not move at any stage. Kew 2024 names pretraining exposure as "the main limitation."
- **Instruction tuning can *lower* language-knowledge scores.** NorEval found Norwegian language-knowledge scores fell in all five base-vs-instruct pairs (for example, AI-Sweden/Llama-3-8B fell from 51.0 to 16.1). Gao 2024 found instruction tuning lowered factual-knowledge scores. Holmström 2023 found GPT-SW3 perplexity on Swedish news rose from 1.92 to 2.66 after SFT on translated data. Any SE.LLMA post-training run should report base and post-trained scores on the same exam with the same prompt format.
- **A little Swedish data goes a long way for behaviour.** About 40 multilingual examples (Shaham 2024) or about 400 Finnish examples (Zosa 2025) gave most of the instruction-following gain. A large translated SFT set is not needed for this purpose and may hurt fluency (Samuel 2026).
- **If SE.LLMA does post-train, prefer on-policy methods and avoid translated text.** Native judges preferred on-policy RL over SFT on translated data 67.5% to 32.5% for Norwegian (Samuel 2026, ICLR). This is the strongest Nordic, peer-reviewed result on post-training method.
- **RL with verifiable rewards mostly sharpens what the base model can already do.** Yue et al. (NeurIPS 2025) found pass@256 unchanged or slightly lower after RL. ProRL and Raghavendra et al. argue otherwise, but both are preprints on English reasoning. In a target language, RL also tends to drift the reasoning back to English (DeepSeek-R1, Park 2025 preprint).
- **Distillation from a stronger teacher is the post-training route with the best evidence of adding ability to a small model.** DeepSeek-R1 distillation beat RL on Qwen-32B on every reported benchmark (AIME 72.6 vs 47.0). Yue et al. say distillation "can introduce new reasoning patterns." No source here tests distillation for Swedish knowledge specifically, so this is an inference, not a finding.
- **Careful wording for the note:** "Published evidence suggests that SFT, DPO and RL mainly improve instruction following and fluency in the target language. Gains on knowledge-heavy multiple-choice tests are small. The best-documented way to add Swedish knowledge is continued pretraining on Swedish text, optionally followed by light post-training or distillation."

---

## Could not verify

- **Poro 2 (LumiOpen/AMD) model cards.** A search snippet reported Finnish MT-Bench 6.75 for Poro 2 8B Instruct vs 4.10 for Llama 3.1 8B Instruct, and IFEval-fi 66.54 vs 47.31. I did not open the model card. These are developer self-reports with an LLM judge.
- **EuroEval / ScandEval Swedish leaderboard (euroeval.com).** The page loads data dynamically and returned no numbers. No base-vs-instruct comparison from the leaderboard is included.
- **GPT-SW3 instruct, Viking, and AI Sweden Llama-3 instruct model cards.** Not opened. No reported base-vs-instruct gains verified (except the AI-Sweden/Llama-3-8B pair as measured by NorEval, entry 11).
- **Diabase Europa 9B and a "BonLM 8B" paper.** These were mentioned in search snippets (MMLU-sv 63.75 → 63.20). Not opened.
- **Chen et al. 2024, "Monolingual or Multilingual Instruction Tuning: Which Makes a Better Alpaca" (EACL Findings).** Not opened.
- **"Evaluating Large Language Models with Human Feedback: Establishing a Swedish Benchmark" (arXiv 2405.14006).** Not opened.
- **"GRPO Beyond English" (alphaxiv 2608.13698) and AdaMame-GRPO (arXiv 2606.15080).** Seen only as search snippets. Not opened.
- **ProRL full-text numbers.** Only the abstract was read.
- **Shaham 2024 exact win rates and Kew 2024 exact scores.** These appear only in figures, which could not be read.
- **Nature page for DeepSeek-R1.** It redirected to a login. Content was verified through the arXiv version only.
