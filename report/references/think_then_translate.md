# Think in English (or with a strong model), then translate: evidence for and against

Question: should SE.LLMA use a system where a strong, often English-dominant model does the reasoning and a smaller model only translates the result into Swedish?

All sources were opened on 2026-11-19. Quotes are verbatim. Some long abstracts were read in segments, so "..." marks a gap between two verbatim segments. Labels:

- **SUPPORTS** means the source finds that the pipeline (or reasoning/working in English) helps.
- **AGAINST** means the source finds that it hurts, or shows a limit that matters for Swedish.
- **MIXED** means the source finds both.

Preprints are flagged. "Venue not listed" means the page I opened shows no venue. It does not mean the paper was never published.

---

## Verified references

### A. Reasoning or working in English, then answering

**1. Etxaniz, Azkune, Soroa, Lopez de Lacalle, Artetxe (2024). "Do Multilingual Language Models Think Better in English?" NAACL 2024 (Volume 2: Short Papers), pages 550–564.**
URL: https://aclanthology.org/2024.naacl-short.46 (preprint: https://arxiv.org/abs/2308.01223). Accessed 2026-11-19.
Label: **SUPPORTS** (with limits)

- "Translate-test is a popular technique to improve the performance of multilingual language models."
- "Experiments over 5 tasks show that self-translate consistently outperforms direct inference"
- Conclusion: "demonstrating that language models are unable to leverage their full multilingual potential"
- Average accuracy over the 5 tasks, from Tables 1 and 4 of the arXiv HTML version (https://ar5iv.labs.arxiv.org/html/2308.01223). Columns are direct inference, then self-translate, then NLLB MT translate-test:

  | Model | Direct | Self-translate | NLLB MT |
  |---|---|---|---|
  | XGLM 0.6B | 39.6 | 39.9 (+0.3) | 43.2 |
  | XGLM 7.5B | 43.4 | 45.7 (+2.3) | 47.6 |
  | LLaMA 7B | 40.6 | 43.4 (+2.8) | 50.7 |
  | LLaMA 30B | 43.1 | 46.6 (+3.5) | 54.2 |

- "Self-translate is more effective for high-resource languages and large models."
- "MT outperforms self-translate, but the gap narrows at scale."
- What this means here: translating into English gave gains of 0.3–3.5 points with the model's own translation. A dedicated MT system gave about 4–11 points. These are classification and reasoning benchmarks. The gains are small for small models. This is the "translate in" direction only. It does not test translating the answer back out.
- Caveat: the numbers come from the arXiv v1 HTML. The NAACL PDF could not be text-extracted here.

**2. Huang, Tang, Zhang, Zhao, Song, Xia, Wei (2023). "Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting." Findings of EMNLP 2023.**
URL: https://arxiv.org/abs/2305.07004. Accessed 2026-11-19.
Label: **SUPPORTS**

- "XLT is a generic template prompt that stimulates cross-lingual and logical reasoning skills"
- "We conduct comprehensive evaluations on 7 typical benchmarks related to reasoning, understanding, and generation tasks"
- "XLT brings over 10 points of average improvement in arithmetic reasoning". The abstract claims a similar gain for open-domain question answering.
- What this means here: the largest gains are on math and on general (not Sweden-specific) question answering.

**3. Shi, Suzgun, Freitag, Wang, Srivats, Vosoughi, Chung, Tay, Ruder, Zhou, Das, Wei (2022). "Language Models are Multilingual Chain-of-Thought Reasoners." arXiv:2210.03057. Venue not listed on the arXiv page.**
URL: https://arxiv.org/abs/2210.03057. Accessed 2026-11-19.
Label: **SUPPORTS** (on math)

- "We introduce the Multilingual Grade School Math (MGSM) benchmark, by manually translating 250 grade-school math problems"
- PaLM-540B average MGSM accuracy, from Table 3 (https://ar5iv.labs.arxiv.org/html/2210.03057): Direct 18.6%, Native-CoT 48.1%, EN-CoT 51.3%, Translate-EN 55.0%.
- "In our best setting, PaLM achieves an average solve rate of 55% across languages."
- What this means here: translating to English beat reasoning in the native language by 6.9 points (55.0 vs 48.1) on grade-school math. Reasoning in English (EN-CoT) beat native reasoning by 3.2 points (51.3 vs 48.1). Math is the kind of task where little knowledge is tied to one language.

**4. Artetxe, Goswami, Bhosale, Fan, Zettlemoyer (2023). "Revisiting Machine Translation for Cross-lingual Classification." arXiv:2305.14240. PREPRINT (venue not listed).**
URL: https://arxiv.org/abs/2305.14240. Accessed 2026-11-19.
Label: **MIXED**

- The abstract says that a stronger MT system, plus less mismatch between training on original text and testing on machine-translated text, makes translate-test "substantially better than previously assumed."
- It also finds that the best approach depends on the task, because different tasks are hit by different sources of cross-lingual transfer loss.
- What this means here: translate-test is a strong baseline for classification, but its value depends on the task.

**5. Liu, Zhang, Zhao, Luu, Bing (2025). "Is Translation All You Need? A Study on Solving Multilingual Tasks with Large Language Models." NAACL 2025.**
URL: https://arxiv.org/abs/2403.10258 (full text: https://arxiv.org/html/2403.10258v3). Accessed 2026-11-19.
Label: **MIXED**. This is the closest test of the full pipeline: translate the query to English, answer in English, translate the answer back.

- "translation into English can boost the performance of English-centric LLMs on NLP tasks" ... "it is not universally optimal."
- "For culture-related tasks that need deep language understanding, prompting in the native language proves more effective."
- Pipeline used for real user queries: the queries were "translated queries with Google Translate API" and "we translate the output back to the original language for consistency."
- Gains on NLP tasks: ChatGPT's high-resource average rose from 50.8 (native) to 61.4 (Google-translate pipeline). Its low-resource average rose from 40.8 to 59.0. On MGSM, ChatGPT's high-resource accuracy was 74.7 with translation vs 44.4 native.
- Losses on culture tasks (M3Exam): "The translation hurts Chinese performance, highlighting the significant influence of translationese on comprehension." This was found for Qwen1.5-72B-Chat and Yi-34B-Chat. The exception was Llama-2-70B-Chat: "Translation helps Llama-2-70B-chat in all the languages."
- Real user queries (ShareGPT), judged by GPT-4o: "For high-resource languages like Japanese, Chinese, and Spanish, original queries have a higher win rate." This was for ChatGPT. Separately, "30% to 74% of queries per language require cultural knowledge."
- What this means here: the pipeline helps weaker, English-centric models on NLP benchmarks. It does not help a model that is already good in a high-resource language on real queries, and it hurts on culture-heavy exams. The M3Exam and win-rate numbers appear only in figures, so I quote the text, not exact values.

**6. Yong, Adilazuarda, Mansurov, Zhang, Muennighoff, Eickhoff, Winata, Kreutzer, Bach, Aji (2025). "Crosslingual Reasoning through Test-Time Scaling." arXiv:2505.05408. PREPRINT.**
URL: https://arxiv.org/abs/2505.05408. Accessed 2026-11-19.
Label: **MIXED**

- Scaling inference compute for English-centric reasoning models "improves multilingual mathematical reasoning across many languages including low-resource languages."
- The authors observe poor generalization "in particular from STEM to cultural commonsense knowledge," even for English.
- "further work is needed to improve reasoning in low-resource languages and out-of-domain contexts."
- What this means here: an English-reasoning model transfers well on math. It does not transfer well to cultural knowledge.

### B. Costs and limits: translation errors, translationese, local knowledge

**7. Singh, Romanou, Fourrier, Adelani, et al. (24 authors), incl. Ruder, Hooker (2024/2025). "Global MMLU: Understanding and Addressing Cultural and Linguistic Biases in Multilingual Evaluation." arXiv:2412.03304 (v2, Feb 2025). PREPRINT (venue not listed).**
URL: https://arxiv.org/abs/2412.03304. Accessed 2026-11-19.
Label: **AGAINST** (translated material distorts, and much knowledge is local)

- "28% of all questions requiring culturally sensitive knowledge."
- "for questions requiring geographic knowledge, an astounding 84.9% focus on either North American or European regions."
- "translation often introduces artefacts that can distort the meaning or clarity of questions in the target language."

**8. Romanou, Foroutan, Sotnikova, Chen, Nelaturu, et al. (59 authors) (2024). "INCLUDE: Evaluating Multilingual Language Understanding with Regional Knowledge." arXiv:2411.19799. PREPRINT on the arXiv page (venue not listed there).**
URL: https://arxiv.org/abs/2411.19799. Accessed 2026-11-19.
Label: **AGAINST** (regional knowledge is the bottleneck)

- Multilingual benchmarks often "translate English resources, ignoring the regional and cultural knowledge" that deployment needs.
- The benchmark "evaluates multilingual LLMs for performance in the actual language environments where they would be deployed."
- It has 197,243 QA pairs from local exams in 44 written languages.
- What this means here: a strong English model only helps if it actually knows the regional facts. Translating its answer does not add those facts.

**9. Ying, Tang, Zhao, Cao, Rong, Zhang (2025). "Disentangling Language and Culture for Evaluating Multilingual Large Language Models." ACL 2025 (Volume 1: Long Papers), pages 22230–22251.**
URL: https://aclanthology.org/2025.acl-long.1082. Accessed 2026-11-19.
Label: **AGAINST** (moving to English hurts culture-specific questions)

- The abstract names a "Cultural-Linguistic Synergy" effect: "models exhibit better performance when questions are culturally aligned with the language."
- It challenges the idea that LLMs "primarily trained on English data" perform the same across languages.
- What this means here: a question about Swedish culture may be answered better in Swedish than after translation into English.

**10. Zhang, Fei, Mohamed, Almeida Carneiro, Konomi, Geng, Asaad, Shang, Vazirgiannis (2026). "The Masked Advantage: Uncovering Local-Language Access to Cultural Knowledge in LLMs." arXiv:2606.07422. PREPRINT.**
URL: https://arxiv.org/abs/2606.07422. Accessed 2026-11-19.
Label: **MIXED** (English wins on generic questions, the local language wins on local knowledge)

- "Across 13 locales and roughly 80 models, we find a consistent English advantage on culture-agnostic questions."
- After adjusting for language proficiency, "local languages show a positive knowledge-access advantage in nearly all locale-model settings."
- "weaker local-language performance does not necessarily imply weaker cultural knowledge."

**11. Thellmann, Stadler, Färber, Lehmann (2026). "Quantifying the Impact of Translation Errors on Multilingual LLM Evaluation." arXiv:2605.24904. PREPRINT.**
URL: https://arxiv.org/abs/2605.24904 (opened via https://www.alphaxiv.org/abs/2605.24904.md). Accessed 2026-11-19.
Label: **AGAINST** (translation errors cost accuracy)

- Target-side translation errors are "consistently associated with measurable, percentage-point drops in translated accuracy".
- From the paper's contributions list: drops of "about 6–8 pp (full model)" and "6–11 pp among English-solvable items."
- What this means here: every translation step in a pipeline can lose several points of correctness, even when the English answer was right.

### C. Swedish specifics

**12. Kunz, Jarochenko, Bollmann (2026). "A Dataset for Probing Translationese Preferences in English-to-Swedish Translation." arXiv:2603.08450. To appear at LREC 2026. PREPRINT.**
URL: https://arxiv.org/abs/2603.08450. Accessed 2026-11-19.
Label: **AGAINST** (small translator models produce translationese in Swedish)

- When smaller Swedish and multilingual LLMs were tested, "we find that they often favor the translationese phrasing."
- "Human alternatives are chosen more often when the English source sentence is omitted" ... "indicating that exposure to the source biases models toward literal translations"
- "even without context models often prefer the translationese variant."
- What this means here: this is direct evidence for the translator step of the proposed design. A small model translating from English into Swedish tends to produce literal, non-idiomatic Swedish.

**13. Kunz (2026). "A Diagnostic Benchmark for Sweden-Related Factual Knowledge." arXiv:2510.21360. To appear at LREC 2026. PREPRINT.**
URL: https://arxiv.org/abs/2510.21360. Accessed 2026-11-19.
Label: **AGAINST** (Sweden-specific knowledge favours Swedish-trained models)

- "Many Swedish benchmarks are translations of US-centric benchmarks and are therefore not suitable for testing knowledge"
- "smaller models with stronger Swedish coverage perform comparably to a multilingual model three times larger"
- "continued pre-training on Swedish generally improves factual knowledge". The abstract also notes partial forgetting of previously known information.
- What this means here: on questions about Sweden, a large English-dominant "thinker" is not automatically better than a smaller model with good Swedish coverage.

**14. Kurfalı, Zahra, Gogoulou, Dürlich, Carlsson, Nivre (2025). "SweSAT-1.0: The Swedish University Entrance Exam as a Benchmark for Large Language Models." NoDaLiDa/Baltic-HLT 2025, pages 331–339.**
URL: https://aclanthology.org/2025.nodalida-1.36. Accessed 2026-11-19.
Label: **MIXED** (context for Högskoleprovet-type tasks)

- "The current version of the benchmark includes 867 questions across six different tasks,"
- "We find that some widely used open-source and commercial models excel in verbal tasks" ... "but we also see that all models, even the commercial ones, struggle with reasoning tasks in Swedish."
- The benchmark is "free from any translation biases."
- What this means here: the reasoning (quantitative) parts are where a strong thinker model could help. The verbal parts test Swedish itself.
- My own inference, not a claim of the paper: tasks such as ORD (word meaning), MEK (sentence completion) and the language parts of LÄS test the Swedish language. Translating them to English changes or removes what is being tested. A Swedish synonym question has no faithful English version.

### Not the same thing: speculative decoding

**Leviathan, Kalman, Matias (2023). "Fast Inference from Transformers via Speculative Decoding." ICML 2023 (Oral).**
URL: https://arxiv.org/abs/2211.17192. Accessed 2026-11-19.

- "In this work we introduce speculative decoding - an algorithm to sample from autoregressive models faster" ... "without any changes to the outputs"
- How it differs: here a small "draft" model guesses tokens and the large model checks them. It is only a speed trick. The output distribution is the large model's own. No translation happens and no Swedish ability is added. It should not be confused with "large model thinks, small model translates," and it does not count as evidence for or against that design.

---

## What this means for SE.LLMA

- **The pipeline helps most where the task is language-neutral reasoning.** On math, gains are large: Translate-EN 55.0% vs Native-CoT 48.1% for PaLM-540B on MGSM (Shi et al.), and "over 10 points" in arithmetic for XLT (Huang et al.). On general NLP benchmarks, ChatGPT went from 50.8 to 61.4 (Liu et al.). For SweSAT's quantitative and logic sections, where "all models ... struggle with reasoning tasks in Swedish" (Kurfalı et al.), a strong model reasoning in English is a reasonable design to test.
- **It cannot do tasks that test Swedish itself.** ORD, MEK and the language-sensitive parts of LÄS ask about Swedish words, idioms and phrasing. Translating them into English removes the thing being tested. This is my inference. No source above measures translate-test on SweSAT verbal items. For these tasks, the Swedish ability has to sit in the model that reads the Swedish.
- **It does not fix the knowledge gap the research note found. It moves it.** A large English-dominant model knows more general facts. For Sweden-specific knowledge, though, "smaller models with stronger Swedish coverage perform comparably to a multilingual model three times larger" (Kunz 2026). Culture-specific questions are answered better in the matching language (Ying et al., ACL 2025; Zhang et al. 2026). Liu et al. found that translation hurt two strong models on a culture-heavy exam.
- **The translator step has its own measurable cost.** Small models translating English into Swedish "often favor the translationese phrasing" (Kunz, Jarochenko, Bollmann 2026). Target-side translation errors cost "about 6–8 pp" of accuracy (Thellmann et al. 2026). For a model that is already strong in the target language, the full translate-answer-translate-back pipeline lost to answering directly on real high-resource-language queries (Liu et al., ChatGPT). Swedish is a mid-resource language, so the outcome is model-dependent and must be measured. Cost also rises, because each query needs a large-model call plus a translation call.
- **Suggested use.** Route by task type. Send language-neutral reasoning (math, logic, data interpretation) through a strong thinker, then a Swedish translator or rewriter. Keep Swedish-language and Sweden-knowledge tasks in a model with strong native Swedish. Evaluate the pipeline against direct Swedish answering on a native, untranslated Swedish benchmark (SweSAT-1.0, the Sweden-facts benchmark) rather than on translated benchmarks, which Global MMLU and INCLUDE show are biased. Rate the translator's Swedish fluency separately from correctness.

---

## Could not verify

- **Exact numbers from the NAACL 2024 PDF of Etxaniz et al.** The PDF downloaded, but text extraction failed (no pdftotext or pypdf on this machine). The numbers above come from the arXiv v1 HTML (ar5iv) and may differ slightly from the camera-ready version.
- **Exact M3Exam accuracies and ShareGPT win rates in Liu et al.** These appear only in figures (Figures 5 and 6), so only the text statements are quoted.
- **A published study of "large model drafts, small model translates/post-edits" for answer generation.** One search found no paper that tests exactly this cascade with a separate small translator. Liu et al. (Google Translate / NLLB-200-3.3B as translator) is the closest. Search results also mentioned LangMark (arXiv:2511.17153) and a Welocalize post-editing study. I did not open them, so they are not cited.
- **Secondary-summary claims about Yong et al. 2025** (e.g. a 3B-parameter threshold, and German/French/Spanish being near English while Bengali/Swahili/Telugu drop) came from search snippets, not the paper. They are not used above.
- **Other culture benchmarks seen only in search results** were not opened: MultiNRC (arXiv:2507.17476), CaLMQA (arXiv:2406.17761), CulturalBench (arXiv:2410.02677), MSQA. They are not cited.
- **English-to-Swedish MT quality numbers for specific small models** (e.g. FLORES or WMT scores) were not found in the time available.
