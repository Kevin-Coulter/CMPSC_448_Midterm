# Who Wrote It? Identifying LLM Families from Their Responses

CMPSC 448 project. The project trains text classifiers to identify which model family (**Claude**, **GPT**, or **Qwen**) generated a chat response. It addresses two research questions:

- **RQ1:** Can the generating model family be identified from the response text?
- **RQ2:** Does including the user's prompt improve identification?

A CNN and a bidirectional LSTM were each trained under three input conditions: input only, output only, and input + output. All six configurations completed 15 epochs, and each selected checkpoint was evaluated once on the held-out test split.

All code is in [`who_wrote_it.ipynb`](who_wrote_it.ipynb). The experimental design and the rationale for each decision are in [`DESIGN_DECISIONS.md`](DESIGN_DECISIONS.md). The final report, with the detailed results and analysis, will be added to this repository.

## Results

Test split: 198 examples (66 per family). The selected epoch has the best validation macro F1; test results were not used for selection.

| Model | Input condition | Selected epoch | Test accuracy | Test macro F1 |
| --- | --- | --- | --- | --- |
| CNN | Input only | 13 | 0.3636 | 0.3579 |
| CNN | Output only | 14 | 0.7879 | 0.7856 |
| CNN | Input + output | 15 | 0.7475 | 0.7447 |
| BiLSTM | Input only | 1 | 0.3889 | 0.3888 |
| BiLSTM | Output only | 2 | 0.6566 | 0.6486 |
| BiLSTM | Input + output | 9 | 0.5556 | 0.5542 |

Values are from the saved output of the notebook's test evaluation cell. On this dataset and split, output only performed best for both architectures; the CNN output-only model reached 78.79% accuracy and 0.7856 macro F1. Adding the prompt did not improve on output only, and input only stayed near chance (0.333) for both models. Each configuration was trained once with one seed, so run-to-run variability is not measured. Confusion matrices are printed in the notebook; the analysis is in the final report.

## Repository contents

| File | Purpose |
| --- | --- |
| `who_wrote_it.ipynb` | Data preparation, models, benchmark, training, test evaluation, and visualizations, with the saved outputs of the completed Colab run |
| `split_reference.csv` | Reference of the selected examples and split assignments; required for every run |
| `DESIGN_DECISIONS.md` | Experimental design and rationale |
| `outputs/progress_log.txt` | Progress log of an early local preprocessing run (not the training log) |

Checkpoints, training histories, test predictions, confusion-matrix files, and figures were written to Google Drive during the Colab run and are **not** included in the repository.

## Dataset

Source: [`lmarena-ai/arena-expert-5k`](https://huggingface.co/datasets/lmarena-ai/arena-expert-5k), with 5,128 rows. Each row is one expert vote comparing two models' responses. User prompts are licensed CC-BY-4.0; model outputs are governed by the respective providers' terms.

### Preparation

- **Parsing.** The conversation fields are stored as text representations of numpy object arrays. They are parsed with a tokenizer-based parser that never executes dataset content and checks that all text is preserved exactly. All 5,128 rows parse.
- **Examples.** Each example is the **final** user–assistant pair of `conversation_a` or `conversation_b`. That response is attributable to the row's `model_a` or `model_b`. Earlier pairs and history turns are excluded because their generating models are not recorded.
- **Filtering.**
  - Only rows whose `language` label is exactly `en` are used. This relies on the publisher's row-level annotation.
  - Only the 38 identifiers in the explicit identifier-to-family mapping are used: 13 Claude, 12 GPT, and 13 Qwen. OpenAI o-series models, `gpt-oss`, `qwq-32b`, Qwen vision models, and one identifier with an `-invalid` suffix are excluded.
  - Pairs with an empty prompt or response are excluded.
- **Deduplication.** Exact duplicate (model, prompt, response) triples are merged. This leaves 2,566 examples: Claude 1,138, Qwen 774, GPT 654.

### Grouping, balancing, and split

- **Grouping.** Rows that share any exact user prompt, anywhere in their history or current turns, are connected transitively. This keeps related conversations and repeated prompts within one split. The 5,128 source rows form 4,622 groups.
- **Balancing.** All 654 GPT examples are kept. 654 Claude and 654 Qwen examples are sampled with seed 42, for 1,962 examples in total. Downsampling discards 604 usable examples.
- **Split.** Whole groups are assigned to the splits by a deterministic greedy procedure (seed 42) that uses only example counts and family labels. No group or user prompt crosses splits.

| Split | Claude | GPT | Qwen | Total |
| --- | --- | --- | --- | --- |
| Train | 522 | 522 | 522 | 1,566 |
| Validation | 66 | 66 | 66 | 198 |
| Test | 66 | 66 | 66 | 198 |

Every run rebuilds the split and compares it with `split_reference.csv`. A missing file stops the notebook. A new reference is written only when `INITIALIZE_SPLIT_REFERENCE = True`.

## Input conditions and models

| Condition | Model input |
| --- | --- |
| Input only | User prompt |
| Output only | Model response |
| Input + output | Prompt, `<sep>`, response |

**Text representation.**
- **Tokenizer:** a regular-expression tokenizer that splits text into word sequences and individual punctuation characters, with each line break as a `<nl>` token.
- **Case and whitespace:** case is preserved; spaces and indentation are discarded.
- **Vocabulary:** built from training text only, with no size cap (90,131 entries including special tokens).
- **Length:** sequences are not truncated.

**Models.**
- **CNN:**
  - 128-dimensional trainable embeddings;
  - convolutions with windows 2, 3, 4, and 5 and 128 filters each;
  - ReLU, then global max pooling over non-padding positions;
  - dropout 0.5 and a three-class linear classifier.
- **Bidirectional LSTM:**
  - 128-dimensional trainable embeddings;
  - one layer with 128 hidden units per direction over packed sequences;
  - the final forward and backward states are concatenated;
  - dropout 0.5 and a three-class linear classifier.

## Training and evaluation

- Six configurations: two models × three input conditions, all using the same examples and split.
- Batch size 16, padded per batch. 15 epochs. Adam with learning rate 0.001. Ordinary cross-entropy without class weights. Seed 42, reset for each configuration.
- Validation loss, accuracy, and macro F1 are recorded every epoch. The best checkpoint has the highest validation macro F1; ties go to lower validation loss.
- After all six runs finished, each best checkpoint was evaluated once on the test split, producing accuracy, macro F1, and a confusion matrix. Test results were not used for checkpoint selection or tuning.
- Hardware: Google Colab, Tesla T4 GPU.

## Viewing the completed results

No execution is needed. Open `who_wrote_it.ipynb` (on GitHub, in VS Code, or in Jupyter) and read the saved outputs: per-epoch logs in the run cell, the results table and confusion matrices in the test evaluation cell, and the figures in the Visualizations section.

## Reproducing the project

Reproducing the results reruns data preparation and trains all six configurations, writing new checkpoints. A GPU runtime is needed for practical run times.

### Local

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python -m pip install datasets jupyter ipykernel
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\jupyter notebook who_wrote_it.ipynb
```

- The first two code cells (Drive mount and Drive output path) are for Colab only and must be skipped locally.
- Outputs go to `outputs/` unless `WHO_WROTE_IT_OUTPUT` is set.
- The dataset may require a Hugging Face login (`hf auth login`), performed outside the notebook.
- Data preparation was run locally; training and evaluation were run on Colab.

### Google Colab (GPU)

1. Open the notebook on a GPU runtime, for example through the Google Colab extension for VS Code. Place `split_reference.csv` next to the notebook.
2. Run the Drive mount cell, then the output-path cell. The output-path cell sets `WHO_WROTE_IT_OUTPUT` to `/content/drive/MyDrive/CMPSC448_WhoWroteIt`.
3. Run the setup cell and the data-preparation cells through the split-reference check.
4. Run the tokenizer, vocabulary count, vocabulary/batching, and model cells. The benchmark cells are optional.
5. Run the training cells: the controls, the helpers, `train_configuration`, the GPU resume self-check, and the run cell.
6. Run the test evaluation cell, then the visualization cells.

On Colab, the training controls cell stops unless `/content/drive` is a mounted file system and the output directory is on it.

### Training controls

All controls are in the **training controls** cell. The notebook is saved with the values used for the final session.

| Control | Saved value | Purpose |
| --- | --- | --- |
| `RUN_CONFIGS` | `ALL_CONFIGS` | Configurations to train; `ALL_CONFIGS` selects all six |
| `STOP_AFTER_EPOCH` | `15` | Last epoch to complete in this execution; `15` completes a run |
| `CONTINUE_EXISTING` | `True` | Must be `True` to continue a run that already has checkpoints |
| `RUN_TEST_EVALUATION` | `True` | Enables the final test evaluation |

With `CONTINUE_EXISTING = False`, a run whose directory already has checkpoints is refused rather than overwritten. Runs already at epoch 15 are skipped.

### Resuming from checkpoints

With `CONTINUE_EXISTING = True`, a run resumes from the epoch after its `latest.pt`. Model, optimizer, and random-number states are restored, and the metric history and best-checkpoint tracking continue. A run is continued only when its saved configuration (including architecture settings and labels) and data identity match.

Before training, the GPU resume self-check saves and reloads a small CUDA training state and confirms that the next step matches; it passed in the saved Colab run. The BiLSTM input + output run was trained for one epoch, then resumed and completed through epoch 15.

### Test evaluation

With `RUN_TEST_EVALUATION = True`, a preflight check first confirms that all six runs have 15 epochs, all required checkpoints, and matching configuration, data identity, vocabulary, and labels. Each `best.pt` is then evaluated once on the test split.

## Generating figures from saved results

The **Visualizations** section at the end of the notebook reads only saved artifacts (`history.json`, `test_confusion_matrix.csv`, `runs/test_results.csv`). It does not train models or run inference. It first checks that all six runs have 15 epochs of history and a 3×3 matrix (Claude, GPT, Qwen) whose total equals the test split size. For each configuration it then shows and saves two figures, 12 in total, as PNG (300 dpi) and PDF in `OUTPUT_DIR/figures/`:

- **Training curves:** loss, accuracy, and macro F1 per epoch for training and validation. A dashed line marks the selected epoch recorded in `test_results.csv`.
- **Confusion matrix:** raw test counts with a color scale shared across the six runs, and row-normalized percentages on a 0–100% scale.

To produce the figures without retraining, run the Drive mount, output-path, and setup cells, then the three visualization cells. This requires the saved run artifacts in the Drive output folder. Matplotlib is used directly; it is preinstalled on Colab.

## Output files

In the completed run, `OUTPUT_DIR` was `/content/drive/MyDrive/CMPSC448_WhoWroteIt` on Google Drive. Paths below are relative to it. **These files are not included in the repository**; the results they hold are preserved in the notebook's saved outputs.

| File | Contents |
| --- | --- |
| `progress_log.txt` | Timestamped progress and per-epoch metrics |
| `runs/<Model>_<condition>/latest.pt` | Checkpoint after the most recent epoch |
| `runs/<Model>_<condition>/best.pt` | Checkpoint with the best validation macro F1 |
| `runs/<Model>_<condition>/epoch_05.pt`, `epoch_10.pt`, `epoch_15.pt` | Fixed-epoch checkpoints |
| `runs/<Model>_<condition>/history.json` | Per-epoch training and validation metrics, timings, and peak GPU memory |
| `runs/<Model>_<condition>/test_predictions.csv`, `test_confusion_matrix.csv` | Test predictions and confusion matrix |
| `runs/test_results.csv` | Test accuracy and macro F1 for all six configurations |
| `figures/<Model>_<condition>_training.png`, `.pdf` | Training and validation loss, accuracy, and macro F1 per epoch, with the selected epoch marked |
| `figures/<Model>_<condition>_confusion.png`, `.pdf` | Test confusion matrix as counts and row-normalized percentages |

Run directory names are `CNN_input`, `CNN_output`, `CNN_input_plus_output`, `BiLSTM_input`, `BiLSTM_output`, and `BiLSTM_input_plus_output`. The benchmark log (`benchmark_log.txt`) was written to the temporary Colab disk and was not preserved.

Each checkpoint contains:
- model and optimizer state;
- the completed epoch;
- the best validation score, its loss, and its epoch;
- the configuration and data identity;
- the vocabulary and label mapping;
- the random-number states and the metric history.

The embedding table (about 11.5 million parameters) accounts for most of a checkpoint's size.

## Status

| Component | Status |
| --- | --- |
| Data preparation, grouping, balancing, split | Executed locally; reproduced on Colab (split reference matched) |
| Tokenizer, vocabulary, models | Executed on Colab; padding-invariance check passed |
| Compute benchmark | Executed on a Colab Tesla T4 |
| GPU resume self-check | Executed on Colab; passed |
| Training (six configurations, 15 epochs each) | Completed on Colab |
| Test evaluation | Completed; preflight passed for all six runs |
| Visualizations | Executed on Colab; 12 figures saved to Drive |
| Final report | To be added |

Benchmark on a Colab Tesla T4, input + output, full sequences. Values are seconds per training step and measure compute cost only, not classification performance:

| Case | CNN | BiLSTM |
| --- | --- | --- |
| Median length, batch 16 | 0.034 | 0.091 |
| 90th-percentile length, batch 16 | 0.062 | 0.258 |
| 99th-percentile length (12,160 tokens), batch 1 | 0.018 | 0.653 |
| Longest (41,447 tokens), batch 1 | 0.049 | 2.204 |

The highest measured GPU memory across these cases was 1.03 GB.

## Limitations

- Results apply to this dataset, its curation, and this split; they do not establish performance on other data or model versions.
- English eligibility relies on the dataset's row-level language label.
- Balancing by downsampling discards 604 usable Claude and Qwen examples.
- Response lengths differ by family (median whitespace word counts before balancing: Claude 318, GPT 574, Qwen 731), so length may act as a shortcut feature.
- The tokenizer discards spaces and indentation, and treats each CJK character run as one token.
- Each configuration is trained once with one seed, so seed variability is not measured. With 198 test examples, small differences between configurations may not be meaningful.
- GPU (cuDNN) LSTM computations may not be bit-for-bit reproducible, including across resumed runs.
- Prompt-only (input-only) performance may reflect dataset associations or selection effects rather than properties of the models.
