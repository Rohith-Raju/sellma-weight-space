# Language ability comes from training data: verified references (collected 2026-11-19)

All sources below were opened today (PDFs downloaded and text-extracted, or pages fetched), and every quote is copied from the source text. The literature supports "the lever is Swedish training data" as the best-evidenced route, but it does not prove that targeted edits cannot work, and two Swedish sources qualify the claim.

## Performance tracks per-language pre-training data

**[Bandarkar2024]** Bandarkar, Liang, Muller, Artetxe, Shukla, Husa, Goyal, Krishnan, Zettlemoyer, Khabsa (2024). The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. ACL 2024. https://aclanthology.org/2024.acl-long.44/ (text read from https://arxiv.org/pdf/2308.16884; accessed 2026-11-19)
- "the performance of a language model in a given language is related to two key factors. (i) First, the amount of pretraining data in the target language. As predicted by the scaling laws (Kaplan et al., 2020), performance in a language increases monotonically with the amount of pretraining tokens. (ii) Second, the cross-lingual transfer happening between languages in the pretraining data and the target language at inference time".
- "while large vocabulary size and balanced pretraining data correlates with highest model performance on medium- and low-resource languages, even English-centric LLMs can go a long way and generalize to over 30 languages."
- **Supports**, with a qualifier: cross-lingual transfer is a second factor, and "These two factors are hard to disentangle".

**[Ahuja2023]** Ahuja, Diddee, Hada, et al. (2023). MEGA: Multilingual Evaluation of Generative AI. EMNLP 2023. https://aclanthology.org/2023.emnlp-main.258/ (text from https://arxiv.org/pdf/2303.12528; accessed 2026-11-19)
- "The multilingual capabilities of these models can be traced to their pre-training data".
- "We observe that for four tasks (PAWS-X, XNLI, XCOPA, and XQuAD) statistically significant positive correlations between the pre-training data size and performance."
- Example given: French 72.1% vs Japanese 67% on PAWS-X (GPT-3.5-Turbo), with "roughly 3.5 B French tokens in the pre-training data versus 214M for Japanese".
- **Supports, qualified by the authors**: GPT-3's language distribution is used as a proxy, and "these two factors correlate well with only a subset of the tasks and what we are measuring is the correlation which might not imply causation."

**[Li2025]** Li, Shi, Liu, Yang, Payani, Liu, Du (2025). Language Ranker: A Metric for Quantifying LLM Performance Across High and Low-Resource Languages. AAAI 2025 (Social Impact Track, per arXiv comment). https://arxiv.org/abs/2404.11553 (accessed 2026-11-19)
- "there is a strong correlation between the LLM's performance in different languages and the proportion of those languages in its pre-training corpus."
- Example: "German, with a proportion of 0.17%, has a high similarity score of 0.723", against languages "with proportions of less than 0.01%".
- **Supports, weakly**: the "performance" measure is an internal-representation similarity to English, not task accuracy.

**[Touvron2023]** Touvron et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv:2307.09288. https://arxiv.org/abs/2307.09288 (accessed 2026-11-19)
- Table 10 (language distribution in pretraining data): en 89.70%, de 0.17%, fr 0.16%, sv 0.15%, no 0.03%, fi 0.03%, da 0.02%.
- "A training corpus with a majority in English means that the model may not be suitable for use in other languages."
- **Supports** as context: Swedish is a small slice even of a major model, though the third-largest identified language in that table.

**[Llama3-2024]** Llama Team, AI @ Meta (2024). The Llama 3 Herd of Models. arXiv:2407.21783. https://arxiv.org/abs/2407.21783 (accessed 2026-11-19)
- "we increased the percentage of non-English data during pre-training to improve the multilingual performance of Llama 3."
- Final data mix includes "8% multilingual tokens"; corpus "about 15T multilingual tokens, compared to 1.8T tokens for Llama 2."
- **Supports**: the developer's own lever for multilingual ability is data.

**[Gemma3-2025]** Gemma Team, Google DeepMind (2025). Gemma 3 Technical Report. arXiv:2503.19786. https://arxiv.org/abs/2503.19786 (accessed 2026-11-19)
- "We also increase the amount of multilingual data to improve language coverage. We add both monolingual and parallel data".
- "We particularly focus on multilinguality in this version, and this directly impacts the quality of our models."
- **Supports.**

## Continued pre-training and language adaptation

**[Zhang2025]** Zhang, Müller-Eberstein, Bassignana, van der Goot (2025). SnakModel: Lessons Learned from Training an Open Danish Large Language Model. NoDaLiDa/Baltic-HLT 2025. https://aclanthology.org/2025.nodalida-1.80/ (text from https://arxiv.org/pdf/2412.12956; accessed 2026-11-19)
- Llama2-7B "which we continuously pre-train on 13.6B Danish words, and further tune on 3.7M Danish instructions."
- Table 3 (ScandEval Danish, test): Llama2-7B base 46.48; SnakModel-7B instruct 56.63 (↑10.15); Llama2-7B base with the same Danish instruction data but no pre-training 49.35.
- Gains "particularly pronounced for sub-tasks based on natural Danish data, including LA (33.43 → 52.91), TM (38.69 → 71.05), and CT (57.05 → 71.88)."
- "a relatively small amount of translated Danish instructions is sufficient to bias models towards generating output in a new language. Nonetheless, … SnakModel-7B instruct, which is trained on non-translated Danish text outperforms the models trained on translated data".
- **Strongly supports**; this is the cleanest ablation isolating continued pre-training, and it separates "output language" (cheap to shift) from competence (needs data).
- **Qualifiers**: "most parameter updates are consolidated in the embeddings, feed-forward up-projections, and language modeling head", so adaptation is somewhat localised by parameter type, though still learned from 13.6B words. Mistral-7B-based models beat it by "up to 4.65% abs.", so base model matters. Viking-7B, trained from scratch on Nordic data, scores only 37.82 in the same table.

**[Kunz2026]** Kunz (2025/2026). A Diagnostic Benchmark for Sweden-Related Factual Knowledge. To appear at LREC 2026 (per arXiv comment). https://arxiv.org/abs/2510.21360 (accessed 2026-11-19)
- Compares AI Sweden's Llama-3 (continued pre-training on Swedish, Norwegian, Danish) with base Llama-3, prompted in Swedish: "165 questions are answered correctly by both models. The original LLaMA answers 92 questions correctly that AI Sweden LLaMA misses, while the latter answers 134 correctly that the original does not."
- "indicating a substantial knowledge gain from CPT, though the dataset remains far from solved."
- "smaller models with stronger Swedish coverage (8B–9B) can perform on par with a much larger multilingual model (27B)."
- "accuracy steadily increases with model size"; "no model answers more than a quarter of the questions correctly."
- **Supports** directly for Swedish: the gap is knowledge, and Swedish data adds it.
- **Qualifier**: "CPT also causes some catastrophic forgetting: AI Sweden LLaMA fails on 92 of the 257 samples (35.80%) that the original model previously answered correctly", "showing limitations of such naive strategies to language adaptation."

**[AISweden-Llama3]** AI Sweden (2024). AI-Sweden-Models/Llama-3-8B model card. Hugging Face. https://huggingface.co/AI-Sweden-Models/Llama-3-8B (accessed 2026-11-19)
- "a continuation of the pretraining process from `meta-llama/Meta-Llama-3-8B`. It was trained on a subset from The Nordic Pile containing Swedish, Norwegian and Danish. The training is done on all model parameters, it is a full finetune."
- "The training dataset consists of 227 105 079 296 tokens"; 92 GPUs for 30 days; 1 epoch.
- **Context only**: the card gives no benchmark gains. It documents the scale of data and compute behind the model evaluated in [Kunz2026] and [Kurfali2025].

**[Ekgren2024]** Ekgren, Cuba Gyllensten, Stollenwerk, Öhman, Isbister, Gogoulou, Carlsson, Casademont, Sahlgren (2024). GPT-SW3: An Autoregressive Language Model for the Scandinavian Languages. LREC-COLING 2024. https://aclanthology.org/2024.lrec-main.695/ (text from https://arxiv.org/pdf/2305.12987; accessed 2026-11-19)
- Trained on "approximately 320B tokens in Swedish, Norwegian, Danish, Icelandic, and English".
- Against GPT-NeoX and Falcon models "which have been trained on a small amount of Swedish data (1B tokens). It is obvious, and perhaps not very surprising, that GPT-SW3 reach the lowest language modeling perplexity on Swedish, Danish and Norwegian data, and that larger models reach lower perplexity."
- **Supports** for language modelling: more Swedish data gives better Swedish perplexity. See [Kurfali2025] for the task-level caveat.

**[Luukkonen2025]** Luukkonen, Burdge, Zosa, Talman, Komulainen, Hatanpää, Sarlin, Pyysalo (2025). Poro 34B and the Blessing of Multilinguality. NoDaLiDa/Baltic-HLT 2025. https://aclanthology.org/2025.nodalida-1.40/ (text from https://arxiv.org/pdf/2404.01856; accessed 2026-11-19)
- "a 34 billion parameter model trained for 1 trillion tokens of Finnish, English, and programming languages".
- "the model outperforms the best FinGPT model already after 100B tokens of training (10%) despite the relatively small proportion of Finnish in the Poro 34B data and the fact that the FinGPT models were trained on 300B tokens in total."
- "the model is benefiting substantially from its training data in other languages even when tested on Finnish tasks."
- **Supports** training as the lever, but **qualifies** "Swedish data only": target-language data combined with large-scale other-language data beat monolingual training.

**[Viking]** LumiOpen / Silo AI (2024). Viking 33B model card. Hugging Face. https://huggingface.co/LumiOpen/Viking-33B (accessed 2026-11-19)
- "pretrained on Finnish, English, Swedish, Danish, Norwegian, Icelandic and code. It is being trained on 2 trillion tokens (1300B billion as of this release)."
- **Context only**: the card has no evaluation numbers. The only verified Viking result I found is the weak 37.82 in [Zhang2025].

**[Csaki2023]** Csaki, Pawakapan, Thakker, Xu (2023). Efficiently Adapting Pretrained Language Models to New Languages. NeurIPS 2023 ENLSP-III workshop. https://arxiv.org/abs/2311.05741 (accessed 2026-11-19)
- "it is effective to add in training data from the new language during pretraining, because those models greatly outperformed the baseline English model. Second, it is better to adapt a pretrained LLM than train a new one from scratch, as the adapted checkpoint performs better on both languages, even though they are trained for one third as long."
- "mixing English data can mitigate the catastrophic forgetting on English".
- **Supports** (Hungarian and Thai).

**[Chau2020]** Chau, Lin, Smith (2020). Parsing with Multilingual BERT, a Small Corpus, and a Small Treebank. Findings of EMNLP 2020. https://aclanthology.org/2020.findings-emnlp.118/ (text from https://arxiv.org/pdf/2009.14124; accessed 2026-11-19)
- "due to the limits of their pretraining data, their benefits do not apply equally to all language varieties."
- "We propose the use of additional language-specific pretraining and vocabulary augmentation … these methods significantly improve performance over baselines, especially in the lowest-resource cases, and demonstrate the importance of the relationship between such models' pretraining data and target language varieties."
- **Supports** (encoder models, dependency parsing).

## Language-specific neurons

**[Tang2024]** Tang, Luo, Huang, Zhang, Wang, Zhao, Wei, Wen (2024). Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models. ACL 2024. https://aclanthology.org/2024.acl-long.309/ (text from https://arxiv.org/pdf/2402.16438; accessed 2026-11-19)
- "LLMs' proficiency in processing a particular language is predominantly due to a small subset of neurons, primarily situated in the models' top and bottom layers."
- Deactivating 1% of neurons: "significantly impairs the generation capabilities in the target language."
- Activating them (Table 4): language accuracy rises, e.g. zh 0.87 → 0.99, fr 0.73 → 0.90, id 0.40 → 0.99. GPT-4-rated content quality moves slightly and not uniformly: zh 4.30 → 4.57, es 3.51 → 4.02, vi 3.70 → 3.57, ja 2.86 → 2.91.
- The authors' own limitation: "our research has only begun to explore the possibility for directing the output language of the model. Developing strategies to harness these identified neurons for enhancing the model's multilingual proficiency is still worth exploring."
- **Qualifies** the conclusion: a small edit can break a language and can select the output language, and the authors call the quality change a "remarkable improvement". What they show is steering of already-learned ability; they do not show added competence.

**[Mondal2025]** Mondal, Sen, Singhania, Jyothi (2025). Language-Specific Neurons Do Not Facilitate Cross-Lingual Transfer. Insights from Negative Results workshop 2025. https://aclanthology.org/2025.insights-1.6/ (text from https://arxiv.org/pdf/2503.17456; accessed 2026-11-19)
- "We find that such neuron-specific interventions are insufficient to yield cross-lingual improvements on downstream tasks (XNLI, XQuAD) in low-resource languages."
- "We fine-tuned the identified language-specific neurons using LoRA but observed no improvement in performance".
- "deactivation at zero significantly increases perplexity (thus degrading generation quality); however, this degradation in perplexity does not directly translate to a decline in task performance."
- **Strongly supports** the "not by a small targeted edit" half. This is the most direct citation for it (Llama 3.1 and Mistral Nemo; a workshop paper).

**[Kojima2024]** Kojima, Okimura, Iwasawa, Yanaka, Matsuo (2024). On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons. NAACL 2024. https://aclanthology.org/2024.naacl-long.384/ (text from https://arxiv.org/pdf/2404.02431; accessed 2026-11-19)
- "tampering with a few language-specific neurons drastically changes the probability of target language occurrence in text generation."
- **Supports** the distinction: the edit changes which language comes out. The paper makes no competence claim.

## Swedish evaluation context

**[Kurfali2025]** Kurfalı, Zahra, Gogoulou, Dürlich, Carlsson, Nivre (2025). SweSAT-1.0: The Swedish University Entrance Exam as a Benchmark for Large Language Models. NoDaLiDa/Baltic-HLT 2025, pp. 331–339. https://aclanthology.org/2025.nodalida-1.36/ (accessed 2026-11-19)

Table 2 average accuracy (867 questions):

| Model | Average |
|---|---|
| GPT-4o | 71.79 |
| GPT-4o-mini | 63.99 |
| Gemma-2-27b | 63.19 |
| Gemma-2-9b | 58.04 |
| Llama-3.1-8B | 46.46 |
| Llama-3-8B | 43.18 |
| AI-Sweden/Llama-3-8B | 40.40 |
| Aya-23-8B | 28.66 |
| GPT-SW3-20b | 22.71 |
| GPT-SW3-6.7b-v2 | 19.15 |
| GPT-SW3-1.3b | 19.03 |

- "multilingual LLMs outperform their Swedish-oriented counterparts".
- On the AI Sweden model: "it achieves slightly lower average accuracy than the original Llama 3 on five of the eight exams. This raises questions on whether continued pre-training on a mix of Scandinavian languages is useful for this task, or whether it may depend on the nature of the selected dataset."
- GPT-SW3 models "frequently select the same options … which highlights potential shortcomings in following instructions."
- Sub-scores cut both ways: the AI Sweden model is higher on vocabulary (ORD 71.25 vs 68.44) and MEK (59.69 vs 55.62), lower on reading (LÄS 56.25 vs 65.00).
- **Mixed.** Bigger and more-trained models do better within each family, but Swedish-adapted models do not. This is the most important qualifier for SE.LLMA: Swedish data alone, without a strong base and instruction tuning, does not win on Högskoleprovet.

**[EuroEval]** EuroEval Swedish leaderboard (successor to ScandEval). Data file https://raw.githubusercontent.com/EuroEval/leaderboards/main/leaderboards/swedish_all_simplified.csv, site https://euroeval.com (file last committed 2026-04-17; accessed 2026-11-19)
- 144 models; rank score, lower is better. The top is all large general multilingual models: claude-sonnet-4-5 (thinking) 1.18, gemini-3-pro-preview 1.19, gemma-4-31B-it 1.22.
- Within Gemma 3 instruct models, rank improves monotonically with size: 27b 1.55, 12b 1.66, 4b 2.34, 1b 3.20, 270m 3.70.
- No Swedish-specific model (GPT-SW3, AI Sweden, KBLab) appears in this file. The Nordic entries are Norwegian (NbAiLab borealis-12b-instruct-preview 1.89, bineric/lynx-instruct-30b 2.03).
- **Supports** "scale and training matter"; **qualifies** "Swedish-specific": the best Swedish scores currently come from large multilingual models.

**[Nielsen2025]** Nielsen, Enevoldsen, Schneider-Kamp (2025). Encoder vs Decoder: Comparative Analysis of Encoder and Decoder Language Models on Multilingual NLU Tasks. NoDaLiDa/Baltic-HLT 2025. https://aclanthology.org/2025.nodalida-1.60/ (text from https://arxiv.org/pdf/2406.13469; accessed 2026-11-19)
- On Danish the top three are decoders, GPT-4 variants then a continued Llama-3-70B: "Similar results were found with Swedish".
- The encoder-vs-decoder gap "varies between languages, but without being correlated to the language resource spectrum."
- **Neutral to mildly qualifying**; this paper does not test data volume.

## Not found

- No KBLab continued-pre-training result was located.
- No peer-reviewed benchmark numbers for Viking exist beyond the SnakModel table.
- EuroEval lists no Swedish-adapted generative model, so there is no leaderboard comparison of adapted vs base for Swedish.
- No paper tests weight edits (ROME/MEMIT-style) for language ability; the "cannot be edited" evidence is the neuron literature above.

## Suggested phrasing for SE.LLMA

"Our results are consistent with the literature: how well a model performs in a language tracks how much of that language (and of transferable related-language text) it saw in training [Bandarkar2024; Ahuja2023], and the documented way to raise target-language competence is further training on native-language text, which in a controlled Danish ablation added about 10 points over the same base model and in Swedish added Sweden-specific facts the base model lacked [Zhang2025; Kunz2026]. Small interventions on language-specific neurons can break a language or switch the output language, but have not been shown to add competence, and targeted neuron fine-tuning gave no downstream gain [Tang2024; Mondal2025]. We therefore conclude that the Swedish gap we measure is best explained as missing knowledge rather than a localised switch, and that Swedish training data is the best-supported lever. This is not a guarantee: existing Swedish-adapted models still trail larger multilingual models on Högskoleprovet, and continued pre-training caused forgetting of about 36% of previously correct answers [Kurfali2025; Kunz2026], so data must be paired with a strong base model, instruction tuning and forgetting mitigation." Avoid the absolute "cannot be targeted": the defensible form is "we found no small edit that does, and prior work has not shown one".

Extracted texts are in a temporary local folder (scratch only, nothing written to the project).