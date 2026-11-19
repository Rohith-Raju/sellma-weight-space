# Fine-tuning changes behaviour vs knowledge: verified references (collected 2026-11-19)

All sources below were opened today (arXiv abstract page plus full-text PDF, and the venue page where one exists); quotes are copied from the fetched text, with PDF line-break hyphens rejoined. None of these papers tests Gemma 3, Swedish, or targeted weight edits, so they support the conclusion only by analogy.

## Verified references

**[zhou2023lima]** Zhou, Liu, Xu, Iyer, Sun, Mao, Ma, Efrat, Yu, Yu, Zhang, Ghosh, et al. (2023). LIMA: Less Is More for Alignment. NeurIPS 2023. https://arxiv.org/abs/2305.11206 ; https://proceedings.neurips.cc/paper_files/paper/2023/hash/ac662d74829e4407ce1d126477f4a03a-Abstract-Conference.html (accessed 2026-11-19)
- Quote (Sec. 2, arXiv v1): "We define the Superficial Alignment Hypothesis: A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users."
- Quote (abstract): "these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output."
- Numbers: 65B LLaMa fine-tuned on 1,000 examples; responses "either equivalent or strictly preferred to GPT-4 in 43% of cases", 58% vs Bard, 65% vs DaVinci003.
- Verdict: **supports**, but it is a hypothesis backed by human preference ratings, not a direct measurement of knowledge.

**[lin2024urial]** Lin, Ravichander, Lu, Dziri, Sclar, Chandu, Bhagavatula, Choi (2024). The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning. ICLR 2024. https://arxiv.org/abs/2312.01552 (accessed 2026-11-19)
- Venue confirmed from the ICLR 2024 proceedings index (https://proceedings.iclr.cc/paper_files/paper/2024); the arXiv PDF itself says "Preprint".
- Quote (PDF abstract): "base LLMs and their alignment-tuned versions perform nearly identically in decoding on the majority of token positions (i.e., they share the top-ranked tokens). Most distribution shifts occur with stylistic tokens (e.g., discourse markers, safety disclaimers). These direct evidence strongly supports the hypothesis that alignment tuning primarily learns to adopt the language style of AI assistants, and that the knowledge required for answering user queries predominantly comes from the base LLMs themselves."
- Numbers (Fig. 1 caption, Llama-2 base vs chat): "On average, 77.7% of tokens are also ranked top 1 by the base LLM (unshifted positions), and 92.2% are within the top 3 (+ marginal)... knowledge-intensive tokens are predominantly found in unshifted positions."
- Verdict: **supports**; this is the closest analogue to "changes how the answer is selected, not what is known".

**[gekhman2024finetuning]** Gekhman, Yona, Aharoni, Eyal, Feder, Reichart, Herzig (2024). Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations? EMNLP 2024 (main, long paper). https://arxiv.org/abs/2405.05904 ; https://aclanthology.org/2024.emnlp-main.444/ (accessed 2026-11-19)
- Quote (abstract): "large language models struggle to acquire new factual knowledge through fine-tuning, as fine-tuning examples that introduce new knowledge are learned significantly slower than those consistent with the model's knowledge. However, we also find that as the examples with new knowledge are eventually learned, they linearly increase the model's tendency to hallucinate... [results] support the view that large language models mostly acquire factual knowledge through pre-training, whereas fine-tuning teaches them to use it more efficiently."
- Quote (Fig. 1 caption): "The best development performance is obtained when the LLM fits the majority of the Known training examples but only few of the Unknown ones."
- Qualifier (footnote 2): "We use the term 'struggle' to describe how LLMs converge slowly for examples containing new factual knowledge." So new facts are eventually fitted; the claim is "slow and harmful", not "impossible".
- Setup: closed-book QA, PaLM 2-S, exact match.
- Verdict: **supports, with a qualifier**.

**[ovadia2024finetuning]** Ovadia, Brief, Mishaeli, Elisha (2024). Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs. EMNLP 2024 (main). https://arxiv.org/abs/2312.05934 ; https://aclanthology.org/2024.emnlp-main.15/ (accessed 2026-11-19)
- Quote (abstract): "while unsupervised fine-tuning offers some improvement, RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge. Moreover, we find that LLMs struggle to learn new factual information through unsupervised fine-tuning, and that exposing them to numerous variations of the same fact during training could alleviate this problem."
- Numbers (Table 2, current-events accuracy; base / base+RAG / fine-tuned / fine-tuned on 10 paraphrases):
  - Mistral 7B: 0.481 / 0.875 / 0.504 / 0.588
  - Llama2 7B: 0.353 / 0.585 / 0.219 / 0.392
  - Orca2 7B: 0.456 / 0.876 / 0.511 / 0.566
- Verdict: **supports, and qualifies**: fine-tuning with paraphrases does add some knowledge (Mistral 0.481 to 0.588).

**[biderman2024lora]** Biderman, Portes, Ortiz, Paul, Greengard, Jennings, King, Havens, Chiley, Frankle, Blakeney, Cunningham (2024). LoRA Learns Less and Forgets Less. TMLR, August 2024 (Featured Certification, per arXiv comments). https://arxiv.org/abs/2405.09673 (accessed 2026-11-19)
- Quote (abstract): "in the standard low-rank settings, LoRA substantially underperforms full finetuning. Nevertheless, LoRA better maintains the base model's performance on tasks outside the target domain... full finetuning learns perturbations with a rank that is 10-100× greater than typical LoRA configurations".
- Quote (intro): "In continued pretraining, the performance gap between full finetuning and LoRA is not closed even with high ranks. In instruction finetuning, on the other hand, high ranks can match full finetuning performance."
- Numbers (math continued pretraining): best LoRA (r=256) peaks at GSM8K=0.203 at 16B tokens; full fine-tuning reaches 0.224 at 4B and 0.293 at 20B.
- Verdict: **supports** "small or low-rank changes are limited for acquiring new domain knowledge"; **qualifies** because full-rank continued pretraining does learn. Domains are code and maths, not factual or linguistic knowledge.

**[ghosh2024closer]** Ghosh, Evuru, Kumar, S, Aneja, Jin, Duraiswami, Manocha (2024). A Closer Look at the Limitations of Instruction Tuning. ICML 2024 (per arXiv comments). https://arxiv.org/abs/2402.05119 (accessed 2026-11-19)
- Quote (abstract): "IT fails to enhance knowledge or skills in LLMs. LoRA fine-tuning is limited to learning response initiation and style tokens, and full-parameter fine-tuning leads to knowledge degradation."
- Quote (abstract): "responses generated solely from pre-trained knowledge consistently outperform responses by models that learn any form of new knowledge from IT on open-source datasets."
- Verdict: **supports** strongly; the scope is instruction tuning on open-source datasets.

**[allenzhu2024physics31]** Allen-Zhu, Li (2024). Physics of Language Models: Part 3.1, Knowledge Storage and Extraction. ICML 2024 (PMLR 235). https://arxiv.org/abs/2309.14316 ; https://proceedings.mlr.press/v235/allen-zhu24a.html (accessed 2026-11-19)
- Quote (abstract): "for knowledge to be reliably extracted, it must be sufficiently augmented (e.g., through paraphrasing, sentence shuffling, translations) during pretraining. Without such augmentation, knowledge may be memorized but not extractable, leading to 0% accuracy, regardless of subsequent instruction fine-tuning."
- Quote (practical implications): "Addressing this during fine-tuning is often too late."
- Verdict: **supports** "the remaining gaps need training, not late edits". It also **qualifies** the wording: failure to answer can mean knowledge is stored but not extractable, so "knowledge gap" is not the only reading of a persistent error. Results come from synthetic biographies.

**[allenzhu2025physics32]** Allen-Zhu, Li (2025). Physics of Language Models: Part 3.2, Knowledge Manipulation. ICLR 2025 (confirmed from https://proceedings.iclr.cc/paper_files/paper/2025). https://arxiv.org/abs/2309.14402 (accessed 2026-11-19; abstract only, full text not opened)
- Quote (abstract): "they cannot efficiently manipulate knowledge from pre-training data, even when such knowledge is perfectly stored in the models, despite adequate training and sufficient model size."
- Verdict: **qualifies**: a wrong answer does not prove missing knowledge.

**[jain2024mechanistically]** Jain, Kirk, Lubana, Dick, Tanaka, Grefenstette, Rocktäschel, Krueger (2024). Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks. ICLR 2024 (confirmed from proceedings index). https://arxiv.org/abs/2311.12786 (accessed 2026-11-19; abstract only)
- Quote (abstract): "(i) fine-tuning rarely alters the underlying model capabilities; (ii) a minimal transformation, which we call a 'wrapper', is typically learned on top of the underlying model capabilities, creating the illusion that they have been modified".
- Verdict: **supports**; synthetic and TinyStories settings.

**[prakash2024finetuning]** Prakash, Shaham, Haklay, Belinkov, Bau (2024). Fine-Tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking. ICLR 2024 (confirmed from proceedings index). https://arxiv.org/abs/2402.14811 (accessed 2026-11-19; abstract only)
- Quote (abstract): "Our findings suggest that fine-tuning enhances, rather than fundamentally alters, the mechanistic operation of the model."
- Verdict: **supports**; a single-task case study.

## Counter-evidence

**[jiang2024instruction]** Jiang, Sun, Shi, Rodriguez, Zhou, Neubig, Lin, Yih, Iyer (2024). Instruction-tuned Language Models are Better Knowledge Learners. ACL 2024. https://arxiv.org/abs/2402.12847 ; https://aclanthology.org/2024.acl-long.296/ (accessed 2026-11-19)
- Numbers: after continued pre-training on new 2023 Wikipedia documents, "the percentage of questions regarding those documents that LLMs answer correctly increases consistently to 27.6%. Subsequent instruction-tuning further improves it to 30.3%"; pre-instruction-tuning gives "17.8% on Llama-2 7B (30.3% → 48.1%) and 16.3% on Llama-2 70B (46.4% → 62.7%)".
- Quote: "the amount of elicited knowledge is still limited, even though the perplexity of documents is minimized".
- Verdict: **contradicts a strong reading** (post-training does add new facts) while showing it is inefficient.

**[mecklenburg2024injecting]** Mecklenburg, Lin, Li, Holstein, Nunes, Malvar, Silva, Chandra, Aski, Yannam, Aktas, Hendry (2024). Injecting New Knowledge into Large Language Models via Supervised Fine-Tuning. arXiv preprint (no peer-reviewed venue found). https://arxiv.org/abs/2404.00213 (accessed 2026-11-19)
- Quote: "Between the base model and the 1x train dataset, the accuracy more than doubles for each of our five post-cutoff events."
- Quote: "In none of the document/dataset-scaling configurations do we outperform RAG with fine-tuning"; SFT got "within 16% (or better) of RAG performance".
- Verdict: **contradicts** "fine-tuning cannot add knowledge" (GPT-4, sports events after the cutoff).

**[gururangan2020dont]** Gururangan, Marasović, Swayamdipta, Lo, Beltagy, Downey, Smith (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. ACL 2020. https://arxiv.org/abs/2004.10964 (accessed 2026-11-19; abstract only)
- Quote (abstract): "a second phase of pretraining in-domain (domain-adaptive pretraining) leads to performance gains, under both high- and low-resource settings."
- Verdict: **qualifies**: further training on more data does add domain competence, which fits "the gaps require training". Covers four domains and eight classification tasks.

**[berglund2024reversal]** Berglund, Tong, Kaufmann, Balesni, Stickland, Korbak, Evans (2024). The Reversal Curse: LLMs trained on "A is B" fail to learn "B is A". ICLR 2024 (confirmed from proceedings index). https://arxiv.org/abs/2309.12288 (accessed 2026-11-19; abstract only)
- Evidence: fine-tuning GPT-3 and Llama-1 on fictitious facts fails in the reverse direction; GPT-4 answers 79% forward vs 33% reverse on real celebrity questions.
- Verdict: **qualifies** both ways: fine-tuning does inject facts in the trained direction, but the result is brittle.

## What the literature does and does not license

- **Supported:** small or low-rank post-training changes mainly shift style, format and how existing knowledge is used; factual knowledge mostly comes from pre-training; adding knowledge afterwards is slow, costly, and needs many varied examples or full-rank continued pre-training.
- **Not supported:** "weight changes can never add knowledge" (Jiang, Mecklenburg, Ovadia's paraphrase result and Biderman's full fine-tuning all show gains).
- **Not supported:** "remaining errors are definitely missing knowledge" (Allen-Zhu and Li show knowledge can be stored but not extractable or usable).
- **Coverage:** the papers study instruction tuning, LoRA and continued pre-training on English-centric models, so they do not directly test the report's intervention.

## Suggested phrasing

"Our small targeted weight edits changed how Gemma 3 4B selects among answer options rather than adding Swedish knowledge. This is consistent with prior findings that light post-training mainly alters format and the use of existing knowledge, while factual knowledge is acquired predominantly in pre-training [zhou2023lima; lin2024urial; gekhman2024finetuning; ghosh2024closer]. The remaining errors are therefore most plausibly knowledge (or knowledge-extractability) gaps that edits of this size cannot close, and that would need substantially more training on Swedish data, such as continued pre-training or full fine-tuning with varied examples [biderman2024lora; allenzhu2024physics31; jiang2024instruction; ovadia2024finetuning]."

Avoid "fine-tuning cannot add knowledge" and "proves"; use "consistent with" and "most plausibly".

## Verification caveats

- OpenReview blocked automated access, so the ICLR and TMLR venues come from the ICLR proceedings indexes and arXiv comment fields, not the OpenReview pages.
- LIMA's hypothesis quote is from arXiv v1 ("Preprint. Under review"); the NeurIPS camera-ready PDF was not opened, only its abstract page.
- Ovadia's numbers are from the latest arXiv PDF, not the ACL Anthology PDF.
- Author lists for LIMA and the other long lists are as shown in arXiv metadata, truncated at 12 names; check LIMA's full list before final citation.

Downloaded PDFs and extracted text are in a temporary local folder (files named by arXiv ID). That directory is shared with other agents, and my extraction pass also wrote `<id>.txt` files next to their PDFs; nothing was overwritten.