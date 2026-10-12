# Who Wrote It? Identifying LLMs from Their Responses

Solo CMPSC 448 project by Kevin Coulter. Due October 11, 2026, at 11:59 PM.

This project investigates whether a convolutional neural network (CNN) and a recurrent neural network (RNN) can identify the LLM family responsible for a response, and whether including the user's prompt improves classification.

This document records the experimental design and its rationale. All six configurations have been trained and evaluated under this design; results are summarized in `README.md` and analyzed in the final report.

## Scope and working principles

- Investigate RQ1 and RQ2 only. RQ3 and RQ4 are outside the current scope.
- Use three labels: Claude (Anthropic), GPT (OpenAI), and Qwen (Alibaba).
- Implement one CNN and one bidirectional LSTM, evaluated under three input conditions.
- Keep the implementation centered on a readable Jupyter notebook, `who_wrote_it.ipynb`.
- Keep the project small and understandable. Each design decision must have a rationale. Material design changes are recorded here before implementation.
- Do not add classifiers, experiment grids, extra research questions, or supporting infrastructure without an explicit scope decision.

The goal is a controlled comparison that a solo author can verify and explain.

## Research questions and experiments

**RQ1: Can we identify which LLM generated a response?** Use response text alone to predict its family with the CNN and LSTM.

**RQ2: Does the user's prompt help identify the LLM?** Compare input only, output only, and input + output for both architectures.

| Architecture | Input only | Output only | Input + output |
| --- | --- | --- | --- |
| CNN | User prompt | Model response | Prompt + separator + response |
| Bidirectional LSTM | User prompt | Model response | Prompt + separator + response |

There are six configurations total. The two output-only configurations also answer RQ1; they do not require separate training runs for each research question. Use identical examples and splits in every configuration, and hold each architecture's settings constant across its three input conditions. This makes differences easier to attribute to the available text.

## Dataset

Source: [lmarena-ai/arena-expert-5k](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k).

Load the dataset through Hugging Face Datasets:

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("lmarena-ai/arena-expert-5k")
```

Authentication, if required, happens outside the notebook. Do not store credentials in project files.

### Curation decisions

| Decision | Rationale |
| --- | --- |
| Predict families rather than exact versions | Models within a family may share training data, training strategies, and design choices that produce similar characteristics. This is a working assumption motivating family classification, not a verified claim about proprietary training details. |
| Restrict labels to Claude, GPT, and Qwen | Satisfies the three-family requirement while keeping classification and analysis manageable. |
| Restrict examples to English | Reduces language-related confounding so family predictions are less likely to depend on different language distributions. |
| Treat each prompt–response pair as one example | Retains usable examples from multiple turns rather than discarding later conversation turns. |
| Extract examples only from each row's current turn (`conversation_a`/`conversation_b`) | Historical responses in `full_conversation` carry no generating model identifiers, and sample checks showed earlier turns were produced by different models than the current row's. Later turns remain eligible through their own rows. Historical responses must not become independently labeled examples. |
| "Current pair" means the final user–assistant pair of each `conversation_a`/`conversation_b` | The final response is the one judged in the row's vote, so it belongs to the row's `model_a`/`model_b`; this was confirmed against `full_conversation` for every row. Earlier pairs in these fields are excluded because their generating models are unverified. |
| Exclude current pairs with an empty prompt or response | An empty text gives no authorship signal. Exclusions are counted separately by reason and side (A/B). The empty-array form `array([], dtype=float64)` is recognized by an exact token match, never by evaluating content. |
| Do not exclude a current text pair because of historical images | Images in earlier turns do not affect the current text pair. History is kept for grouping. |
| Rows whose history cannot be reliably parsed remain pending | Without parsed history, a row's split group cannot be determined, so it must not enter extraction with an assumed group. |
| Use an explicit identifier-to-family mapping (listed below) | Makes inclusion auditable. Only the 38 listed identifiers are labeled; every other identifier is excluded. Qualifying dated, thinking, `-old`, and system-prompt variants keep their family label. |
| Exclude o-series (`o3-*`, `o4-mini-*`), `gpt-oss-*`, `qwq-32b`, Qwen vision variants, and `qwen3-235b-a22b-instruct-2507-invalid` | Their names do not clearly match the family naming, and including them could blur family boundaries. The meaning of `-invalid` is undocumented. |
| Include only rows whose `language` label is exactly `en` | Excludes all other labels, including `und` and `<err>`. English eligibility relies on the publisher's row-level annotation, which does not guarantee every final pair is entirely English. |
| Use the immediately preceding user message as input | Gives input-only a clear definition and excludes earlier model responses from that condition. Later prompts may depend on omitted context; document this limitation. |
| Keep family counts as balanced as possible | Reduces the advantage of predicting a more frequent family and makes comparisons easier to interpret. |

### Identifier-to-family mapping

- **Claude (13):** claude-3-5-haiku-20241022, claude-3-5-sonnet-20241022, claude-3-7-sonnet-20250219, claude-3-7-sonnet-20250219-thinking-32k, claude-opus-4-1-20250805, claude-opus-4-1-20250805-thinking-16k, claude-opus-4-1-20250805-thinking-16k-old, claude-opus-4-20250514, claude-opus-4-20250514-thinking-16k, claude-sonnet-4-20250514, claude-sonnet-4-20250514-thinking-32k, claude-sonnet-4-5-20250929-old, claude-sonnet-4-5-20250929-thinking-32k
- **GPT (12):** chatgpt-4o-latest-20250326, chatgpt-4o-latest-20250326-old, gpt-4.1-2025-04-14, gpt-4.1-mini-2025-04-14, gpt-4o-2024-11-20, gpt-4o-mini-2024-07-18, gpt-5-chat, gpt-5-high, gpt-5-high-new-system-prompt, gpt-5-mini-high, gpt-5-nano-high, gpt-5-old
- **Qwen (13):** qwen-max-2025-01-25, qwen3-235b-a22b, qwen3-235b-a22b-instruct-2507, qwen3-235b-a22b-no-thinking, qwen3-235b-a22b-thinking-2507, qwen3-30b-a3b, qwen3-30b-a3b-instruct-2507, qwen3-coder-480b-a35b-instruct, qwen3-max-2025-09-23, qwen3-max-2025-09-26, qwen3-max-preview, qwen3-next-80b-a3b-instruct, qwen3-next-80b-a3b-thinking

Each curated example must retain the family label, original model identifier, user prompt, response, and enough source information to trace its origin and assign its split.

The source contains paired model responses and conversation histories. Models can change between evaluation turns. Label each response only when its generating model can be verified; do not assign the current row's model identifiers to every historical response. Repeated history must not create duplicate training examples.

Model-to-family mappings, usable counts, turn attribution, English filtering, and conversation grouping have been verified from the loaded data. Ambiguous identifiers were resolved explicitly rather than including every OpenAI model in the GPT label. Two limitations remain:
- English filtering relies on the publisher's row-level language label, which does not guarantee that every final pair is entirely English.
- Turn attribution covers only the final pair of each side. Earlier pairs and history turns lack verified generating models and are excluded.

**Balancing.** After curation and deduplication, there were 1,138 Claude, 654 GPT, and 774 Qwen examples. Keep all 654 GPT examples. Randomly sample 654 Claude and 654 Qwen examples without replacement, using seed 42 on a stable example ordering, for 1,962 examples in total. Rationale: equal family representation, simpler interpretation, and lower compute cost. Downsampling discards 604 usable examples (484 Claude, 120 Qwen). Group IDs and source associations are kept from before sampling and are not recomputed.

Document the dataset source, collection method described by its publisher, access date, revision when available, filtering, exclusions, deduplication, and final counts. Attribute the source and follow its published usage terms. Do not republish raw data without checking applicable permissions.

## Splitting and leakage prevention

Target an **80% training / 10% validation / 10% testing** split with seed **42**.

Group assignment: whole represented groups are assigned with one deterministic greedy procedure using seed 42. The decision uses only example counts and family labels, never model performance. The resulting assignment is fixed and reused for all six configurations.

- Keep all examples from the same originating conversation in one split.
- Keep responses to the same prompt together, including repeated prompts across conversations.
- Grouping method: connect rows that share an exact user prompt anywhere in their histories or current turns, then merge connections transitively. Row `id`s identify rows, not conversations, so they cannot be used for grouping. This conservatively prevents related conversation context and repeated prompts from crossing splits. Inspect group sizes before splitting.
- Preserve approximate family balance without breaking these groups.
- Build the vocabulary exclusively from training data.
- Use validation data for checkpoint selection and reserve testing for final evaluation.

Grouping reduces leakage from shared prompts and conversation context. Because groups vary in size, exact split proportions and exact per-split class balance may not be achievable; report the actual counts.

## Text representation

| Setting | Choice and rationale |
| --- | --- |
| Tokenization | Word tokens, yielding shorter sequences than character tokenization and allowing more text within practical compute limits. |
| Text signals | Preserve capitalization, punctuation, and line breaks because they may carry authorship information. |
| Tokenizer | Python standard-library `re`. It matches line breaks (CRLF or LF, each as one dedicated `<nl>` token), word sequences (`\w+`), and individual non-whitespace punctuation characters (`[^\w\s]`). Case is preserved. All other whitespace is ignored, so **spaces and indentation are not preserved**. This is a simple word-and-punctuation tokenizer that keeps selected formatting signals without external dependencies. Special tokens (`<nl>`, `<sep>`, `<pad>`, `<unk>`) cannot collide with text, because the tokenizer always splits `<` and `>` into separate tokens. |
| Input + output sequence | Tokenized prompt, then `<sep>`, then tokenized response. |
| Vocabulary | Learn from training text only; include padding, unknown-word, and input/output separator tokens. This prevents vocabulary leakage and supports unseen words. |
| Embeddings | Train 128-dimensional embeddings from scratch. This keeps the pipeline self-contained while providing useful representational capacity. |
| Combined condition | Concatenate prompt, a dedicated separator token, and response. One network processes the pair, keeping the comparison straightforward. |
| Sequence length | Retain complete sequences. Full-length benchmarks on a Colab Tesla T4 passed for both models, including the longest training sequence (41,447 tokens). The highest measured GPU memory was 1.03 GB. No truncation is applied. |
| Padding | Pad to the longest sequence within each batch. This avoids padding every example to the dataset-wide maximum. |

Any truncation or text exclusion introduced for compute reasons must be documented with the affected example counts. Word tokenization alone does not guarantee that long-range context will be learned.

## Architectures

### CNN

Use a trainable 128-dimensional embedding layer, parallel one-dimensional convolutions with window sizes **2, 3, 4, and 5**, and **128 filters per window size**. Apply global max pooling over valid positions and concatenate the pooled features into a 512-dimensional vector. Apply dropout, then a linear classifier with three outputs.

Different windows capture local patterns at several token lengths. Equal filter counts give each length equal capacity. Global max pooling detects whether a learned pattern appears anywhere in the sequence and produces a fixed-size representation. Padding must be excluded from pooling, and sequences shorter than a convolution window must be handled correctly. Apply **ReLU after each convolution**: it adds a standard nonlinear transformation before max pooling.

### Bidirectional LSTM

Use a trainable 128-dimensional embedding layer and **one bidirectional LSTM layer with 128 hidden units per direction**. Concatenate the final forward and backward hidden states into a 256-dimensional vector, apply dropout, and use a linear classifier with three outputs.

Bidirectionality allows representations to incorporate preceding and following context, which is appropriate because the full text is available at classification time. One recurrent layer provides useful capacity while keeping training and explanation straightforward. Use packed sequences so the LSTM skips padding and its final states correspond to actual text.

Both architectures use **dropout of 0.5 before the classifier** to reduce overfitting. The dimensions are agreed starting values, not evidence that the architectures are optimal.

## Training and checkpoints

| Setting | Value | Rationale |
| --- | --- | --- |
| Optimizer | Adam with default settings, including learning rate 0.001 | Provides a standard starting point without expanding into a tuning search. |
| Loss | Ordinary cross-entropy on the three class logits, without class weights | Matches mutually exclusive multiclass classification. Classes are balanced by sampling, so no weights are needed. |
| Batch size | 16 | Offers a compromise between memory and update frequency. It is not changed automatically: an out-of-memory error stops the run. |
| Epochs | 15 for each configuration | Establishes a fixed, consistent training budget; does not guarantee convergence. |
| Best checkpoint | Highest validation macro F1; ties resolved by lower validation loss | Gives every family equal importance and aligns selection with evaluation. |
| Additional checkpoints | Epochs 5, 10, and 15 | Preserves progress without excessive checkpoint files. |
| Random seed | 42 throughout | Supports repeatable sampling, splitting, initialization, and batch ordering. |
| Repetitions | One run per configuration | Keeps the six-experiment workload within scope. |
| Hardware | Google Colab GPU (Tesla T4) | The CPU compute check was impractical locally, and the T4 benchmarks passed with full sequences. |

Reset the seed before initializing each configuration rather than relying solely on a single seed call at notebook startup. Record software versions, hardware, and any compute-driven adjustments. A fixed seed does not guarantee identical CPU and GPU results, and one run does not measure variability across seeds.

Checkpoints include:
- model and optimizer state;
- the completed epoch;
- the best validation macro F1, its loss, and its epoch;
- the configuration and data/split identity;
- the vocabulary and label mapping;
- the RNG states needed to continue the shuffled training sequence;
- the metric history.

Each configuration has its own directory, containing `latest.pt` (every epoch), `best.pt`, and `epoch_05/10/15.pt`. A run is continued only when explicitly requested, and only if its configuration and data identity match.

The first training session ran only epoch 1 of BiLSTM input + output to measure the cost of mixed-length batches before all runs were launched. That run was then resumed and completed through epoch 15. The GPU resume self-check passed on Colab before training. Evaluate each configuration's best validation checkpoint on the test set. Do not select checkpoints or settings using test results.

## Evaluation and reporting

Report **accuracy**, **macro F1**, and **confusion matrices** for all six configurations. Accuracy measures overall correctness; macro F1 gives each family equal weight; confusion matrices show which families are mistaken for one another.

Present one comparison table for the six configurations. Discuss whether output identifies families, whether input alone predicts family, and whether adding input improves output-only performance. Strong input-only performance may indicate dataset associations or selection effects; it does not demonstrate that the prompt uniquely determines the generating family.

The report must include dataset curation, architecture and training details, results, analysis addressing RQ1 and RQ2, limitations, and lessons learned. Do not claim measured performance before experiments are completed.

## Deliverables and current stage

Submit a GitHub repository containing the code/notebook and a PDF report identifying Kevin Coulter as the individual author.

Completed on Colab (Tesla T4), as recorded in the notebook's saved outputs:
- the GPU resume self-check passed;
- all six configurations completed 15 epochs with the settings above;
- the test preflight passed, and each best validation checkpoint was evaluated once on the test split (accuracy, macro F1, and confusion matrices);
- training-curve and confusion-matrix figures were generated from the saved artifacts.

Checkpoints, histories, predictions, and figures are stored on Google Drive, not in the repository. Remaining work: add the final PDF report to the repository.
