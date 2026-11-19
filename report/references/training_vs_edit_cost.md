# Cost of language training vs a targeted edit: verified references (collected 2026-11-19)

All quotes checked by opening the source on 2026-11-19 (model card via WebFetch; papers via the arXiv PDF, text extracted with pypdf).

**[AISweden-Llama3]** AI Sweden (2024). AI-Sweden-Models/Llama-3-8B, model card. https://huggingface.co/AI-Sweden-Models/Llama-3-8B
- "AI-Sweden-Models/Llama-3-8B is a continuation of the pretraining process from meta-llama/Meta-Llama-3-8B."
- "The training is done on all model parameters, it is a full finetune."
- "The training dataset consists of 227 105 079 296 tokens."
- "The training used 23 nodes of a duration of 30 days, where one node contained 4X Nvidia A100 GPUs, yielding 92 GPUs."
- Derived: 92 GPUs x 30 days x 24 h = 66,240 A100 GPU hours (our arithmetic; the card does not state GPU hours).

**[Zhang2025]** Zhang, Müller-Eberstein, Bassignana, van der Goot (2025). SnakModel: Lessons Learned from Training an Open Danish Large Language Model. NoDaLiDa/Baltic-HLT 2025. https://arxiv.org/abs/2412.12956
- Llama2-7B continuously pre-trained on 13.6B Danish words (see language_ability_training_data.md).
- "SNAKMODEL-7Bbase is trained on private infrastructure with one node, containing four NVIDIA A100-PCIe 40GB GPUs."
- "Total time of training took 8,928 GPU hours (93 days × 24 hours × 4 GPUs) between March–June 2024."
- "This is equivalent to 272.3 kg CO2 eq. emitted".

**[Meng2023-MEMIT]** Meng, Sen Sharma, Andonian, Belinkov, Bau (2023). Mass-Editing Memory in a Transformer. ICLR 2023. https://arxiv.org/abs/2210.07229
- Appendix B.3: "ROME takes 44,248.26 sec ≈ 12.29 hr for 10,000 edits on GPT-J, which works out to approximately 4 seconds per edit."
- "All experiments are run on workstations with NVIDIA A6000 GPUs".
- Ethics section: memory-editing methods "potentially reduce the cost and energy needed to fix errors in a model".
- Caveat: these are single-fact edits on GPT-J (6B); see knowledge_editing_limits.md for what such edits do not achieve.

## How to use
- Supported: language adaptation by continued pre-training costs thousands to tens of thousands of GPU hours (8,928 for a Danish 7B; ~66,000 for AI Sweden's Swedish Llama-3-8B); a single targeted edit costs seconds of GPU time.
- Not a like-for-like comparison: training adds broad ability, an edit changes one behaviour/fact. That asymmetry is the point of the hypothesis.

## Models are often run compressed; compression can undo small weight changes (checked 2026-11-19)

**[Google2025-QAT]** Google Developers Blog (18 Apr 2025). *Gemma 3 QAT Models: Bringing state-of-the-Art AI to consumer GPUs.* https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/
- "To make Gemma 3 even more accessible, we are announcing new versions optimized with Quantization-Aware Training (QAT)"
- "This enables you to run powerful models like Gemma 3 27B locally on consumer-grade GPUs like the NVIDIA RTX 3090."
- "Using int4 means each number is represented using only 4 bits – a 4x reduction in data size compared to BF16."
- "Gemma 3 27B: Drops from 54 GB (BF16) to just 14.1 GB (int4)"

**[Ollama-gemma3]** Ollama library, gemma3. https://ollama.com/library/gemma3/tags
- Tags page: "gemma3:4b" and "gemma3:latest" share ID 68ceab7e468a (3.4GB) with "gemma3:4b-it-q4_K_M".
- Registry manifest config for library/gemma3:4b (https://registry.ollama.ai/v2/library/gemma3/manifests/4b, config blob sha256:4bf7b0a45f1af5d9f6f8b847044d8379eef6b36f3ca8b4dcc63a8dfed7426e53): {"model_format":"gguf","model_family":"gemma3",...,"file_type":"Q4_K_M"}.
- So the default Gemma 3 4B download in Ollama is Q4_K_M, the format we use.

**[Zhang2025-unlearn]** Zhang, Z., Wang, F., Li, X., Wu, Z., Tang, X., Liu, H., He, Q., Yin, W., Wang, S. (2025). *Catastrophic Failure of LLM Unlearning via Quantization.* ICLR 2025. https://arxiv.org/abs/2410.16454
- Abstract: "the unlearned model retains an average of 21% of the intended forgotten knowledge in full precision, which significantly increases to 83% after 4-bit quantization".
- Mechanism (paper body, see methods_models_benchmarks.md): "minimal changes to model weights during unlearning", so "quantization is likely to map the weights of the target LLM and the unlearned LLM to the same values."
- Caveat: unlearning by fine-tuning, not a ROME-style edit; an analogue.
