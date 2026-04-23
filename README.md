# Sexism Detection — NLP Assignments 2025/26

> **Course**: Natural Language Processing · University of Bologna (Master's in AI)

This repository contains the implementation of two NLP assignments focused on **sexism detection** in social media text, using deep learning and large language models. The two assignments are complementary: the first explores classical deep learning architectures, the second investigates the capabilities of modern LLMs through prompting strategies.

---


## Assignment 1 — EXIST 2023 Task 2: Multi-class Sexism Detection

### Task Definition

The objective is to classify tweets according to the **intention** of the author within the [EXIST 2023](https://clef2023.clef-initiative.eu/) shared task framework. Each tweet is assigned to one of four categories:

| Label | Description |
|-------|-------------|
| **Non-sexist** (`-`) | The tweet does not contain sexist content |
| **Direct** | The author expresses sexism directly or incites it |
| **Judgemental** | The author condemns or judges sexist situations |
| **Reported** | The author reports or describes a sexist experience |

Only English-language tweets are used. Labels are aggregated via majority vote from multiple annotators; instances without a clear majority are discarded.

### Dataset & Preprocessing

The dataset is sourced from the EXIST 2023 corpus and organized into `training.json`, `validation.json`, and `test.json` splits.

**Preprocessing pipeline:**
1. Load and parse JSON files into Pandas DataFrames
2. Majority-vote label aggregation; `UNKNOWN` labels are removed
3. Language filtering (`lang == 'en'`)
4. Column reduction to: `id_EXIST`, `lang`, `tweet`, `label`
5. Integer encoding of labels (`-` → 0, `DIRECT` → 1, `JUDGEMENTAL` → 2, `REPORTED` → 3)
6. Text cleaning: emoji removal, hashtag/mention/URL stripping, special character removal, lowercasing, and POS-aware lemmatization

**EDA findings:**
- Tweet lengths are concentrated between **10–50 words** and **100–300 characters**, consistent with the Twitter 280-character limit
- Class priors are stratified across all splits (Train/Val/Test)
- Annotator demographics (gender, age) are uniformly distributed, mitigating labeling bias
- A normalized inter-annotator agreement score confirms overall dataset reliability
- Text length is not a trivial discriminating feature across classes

### Text Encoding

Textual input is encoded using **GloVe** pre-trained word embeddings.

- Vocabulary is built exclusively on the **training set** to prevent data leakage
- Special tokens: `<PAD>` (id 0) and `<UNK>` (id 1)
- OOV tokens are handled via two strategies: **random initialization** (baseline) and **context-based initialization**
- Embedding matrix quality is verified via SVD and UMAP visualizations
- A bias analysis is performed to quantify gendered associations in the GloVe space (relevant given the task's sensitivity to gender)
- Sequence length is set to the **95th percentile** of token counts to balance coverage and computational cost

### Model Architectures

Three neural architectures are implemented and compared:

#### 1. Baseline — BiLSTM
- Embedding layer initialized with GloVe vectors (`mask_zero=True`)
- Single **Bidirectional LSTM** layer
- Dense softmax classification head (4 classes)

#### 2. Stacked — 2-Layer BiLSTM
- Same embedding setup
- First BiLSTM with `return_sequences=True` → second BiLSTM
- Higher capacity for capturing hierarchical dependencies

#### 3. Transformer — Twitter-roBERTa
- Checkpoint: `cardiffnlp/twitter-roberta-base-hate-latest`
- Pre-trained on ~58M tweets and fine-tuned for hate speech detection
- Classification head adapted to 4 labels (`num_labels=4`, `ignore_mismatched_sizes=True`)
- Input tokenized with BPE; padding handled dynamically via `DataCollator`

### Training Strategy

- **Multi-seed evaluation** using seeds `[111, 222, 333]` for statistical robustness
- **Loss**: Sparse Categorical Crossentropy
- **Primary metric**: Macro F1-Score (monitored via a custom `MacroF1Logger` callback)
- Early stopping based on validation Macro F1
- Class weights used to handle label imbalance

### Results

All models are evaluated on the **test set**. Results are reported as **mean ± std** across the three seeds. An ensemble strategy based on prediction confidence (max softmax probability) is also evaluated.

| Model | Macro F1 (Test) | Notes |
|-------|----------------|-------|
| Baseline BiLSTM | — | GloVe + single BiLSTM |
| Stacked BiLSTM | — | GloVe + 2-layer BiLSTM |
| Twitter-roBERTa | — | Fine-tuned Transformer |
| Ensemble | — | Confidence-based selection |

> Exact metric values depend on training runs; refer to the notebook outputs for the full tables.

### Error Analysis

A comprehensive error analysis is performed across all architectures:

- **Ensemble evaluation**: Confidence-based prediction selection across Baseline, Stacked, and Transformer
- **Class-wise error distribution**: Identifies dominant misclassification patterns per ground-truth class
- **Error rate vs. text length**: Investigates whether sequence length correlates with prediction failure
- **Comparative error analysis**: Cross-model comparison of exclusive successes and failures
- **Slang robustness**: Conditional error rate on informal/slang-heavy texts vs. the global baseline
- **OOV impact**: Correlates OOV token density with LSTM error rate (Transformers are immune due to BPE subword decomposition)
- **Precision-Recall curves**: Per-class Macro AP comparison across all architectures
- **High-confidence failures**: Isolates the most severe errors — cases where the model was wrong but highly confident

---

## Assignment 2 — EDOS Task B: Sexism Detection with LLMs

### Task Definition

The task is sourced from the [EDOS (Explainable Detection of Online Sexism)](https://github.com/rewire-online/edos) shared task. Given an input sentence, the model must assign it to one of **five categories**:

| Label | Description |
|-------|-------------|
| **Not sexist** | No sexist content detected |
| **Threats** | Threats or incitement to violence against women |
| **Derogation** | Demeaning or derogatory language |
| **Animosity** | Hostile attitudes toward women as a group |
| **Prejudiced discussion** | Reproducing or reinforcing sexist stereotypes |

### Dataset

- `a2_test.csv` — 300 balanced test samples (balanced across sexist / non-sexist)
- `demonstrations.csv` — 1,000 samples used as the few-shot demonstration pool

Both datasets are sourced from the [NLP course material repository](https://github.com/lt-nlp-lab-unibo/nlp-course-material).

### Models

Two instruction-tuned LLMs are selected for evaluation:

| Model | Parameters | Architecture |
|-------|-----------|--------------|
| **LLaMA-3.1-8B-Instruct** | 8B | Decoder-only Transformer |
| **Mistral-7B-Instruct-v0.3** | 7B | Decoder-only Transformer |

Both models are loaded with **4-bit NF4 quantization** (double quantization, FP16 compute via BitsAndBytes) to fit into a single-GPU environment. Each model is wrapped using its native chat template to structure inputs into system/user/assistant roles.

### Methodology

#### Task 1 — Model Setup
4-bit quantization configuration applied uniformly to both models for a fair and controlled comparison.

#### Task 2 — Prompt Design
A structured prompt template is designed to instruct the model to classify the input text into one of the five target classes. The `prepare_prompts` function handles template population.

#### Task 3 — Inference
Due to GPU memory constraints, inference is performed **sequentially** (batch size = 1) to avoid OOM errors from KV-cache accumulation. A `generate_responses` function handles generation; a `process_response` function parses model outputs into discrete labels.

#### Task 4 — Metrics
Evaluation uses **Macro F1-Score** and **Fail Ratio** (proportion of responses that do not map to any valid class). Failed responses are assigned the `not-sexist` label as a fallback.

#### Task 5 — Few-Shot Prompting Strategies

Multiple few-shot configurations are systematically evaluated:

| Strategy | Description |
|----------|-------------|
| **Zero-Shot** | No demonstrations provided |
| **Baseline Few-Shot** | K examples sampled head-first per class (K = 1, 2, 3, 4, 10) |
| **Mixed Few-Shot** | Interleaved examples per class (alternating class order) |
| **Length-Constrained Mixed** | Short examples selected per class to keep prompts concise |
| **Chain-of-Thought (CoT)** | Explicit step-by-step reasoning instructions inserted before the query |
| **Prompt Verbosity** | Short / Medium / Long CoT templates compared on the best-performing strategy |

#### Task 6 — Error Analysis

A systematic comparative analysis is conducted across all configurations:

- **Zero-Shot vs. Few-Shot**: Performance gap quantification
- **Shot-size impact**: How classification quality scales with the number of demonstrations (1→10)
- **Mixed vs. sequential demonstrations**: Whether example ordering in the prompt affects results
- **CoT effect**: Whether step-by-step reasoning instructions improve class discrimination
- **Prompt verbosity**: Impact of instruction detail level on Macro F1 and Fail Ratio
- **Confusion matrices**: Per-class error patterns for both LLMs
- **Qualitative analysis**: Inspection of generated responses to understand generation quality and instruction-following behavior

### Key Observations

- **LLaMA-3.1** tends to produce more balanced predictions across all five classes, resulting in a higher Macro F1, but also a higher fail ratio in some configurations
- **Mistral-v0.3** shows stronger instruction-following with lower fail rates but can be biased toward specific dominant classes
- Few-shot prompting consistently improves performance over zero-shot; the **mixed interleaving** strategy provides more stable results than sequential class-grouped demonstrations
- **Chain-of-Thought** prompting is most effective at higher shot counts (4–10 shots), where the model has enough context to reason meaningfully
- Length-constrained demonstrations help avoid prompt bloat and generally maintain or improve performance
- Both models struggle most with the **Animosity** and **Prejudiced Discussion** classes, which require deeper contextual understanding

---
