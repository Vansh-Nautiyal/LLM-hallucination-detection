# HaluEval Hallucination Detection

This repository fine-tunes `bert-base-uncased` to classify a HaluEval QA answer
as reference-grounded (`0`, not hallucinated) or benchmark-labeled hallucinated
(`1`). Reusable data preparation and evaluation live in the root `common`
package so later model experiments can share the same examples and source-level
splits.

## Project layout

```text
LLM-Hallucination-Detection/
├── common/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── evaluation.py
├── bert_halueval/
│   ├── README.md
│   └── src/
│       ├── train_bert.py
│       └── predict.py
├── data/
│   ├── qa_data.json
│   └── splits/
│       └── qa_split_seed42.json
└── outputs/
    └── <experiment-name>/
```

## Shared components

- `common/preprocessing.py` loads JSON, wrapped JSON, and JSONL; validates and
  constructs paired answer examples; creates and validates source-level split
  manifests; and tokenizes inputs with knowledge-first truncation.
- `common/evaluation.py` computes class metrics, ROC-AUC, classification
  reports, confusion matrices, prediction CSVs, inference latency, peak CUDA
  memory, and JSON result files. It contains no model-specific code.
- `bert_halueval/src/train_bert.py` handles BERT model/tokenizer setup, dynamic
  batch padding, training arguments, validation-based checkpoint selection,
  final test inference, and experiment measurements.
- `bert_halueval/src/predict.py` is the interactive inference app. It loads the
  saved model directly and does not import training code.

## Environment setup

From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

For the project's NVIDIA GPU configuration, install a CUDA-enabled PyTorch
build after the other requirements:

```powershell
python -m pip install --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

Training automatically selects CUDA when available. `--precision auto` uses
FP16 on CUDA and full precision on CPU. Use `--precision no`, `--precision
fp16`, or supported `--precision bf16` to choose explicitly.

## Split manifests

Source IDs refer to original source-record positions; the correct and
hallucinated answer for one record always stay in the same partition. The
manifest records the train/validation/test source IDs, seed, proportions,
format version, dataset fingerprint, and usable source IDs. A fingerprint or
split mismatch is an error; the program never silently replaces an existing
incompatible manifest.

The default manifest is `data/splits/qa_split_seed42.json`. By default,
`--split-manifest-mode auto` creates it if missing and reuses it if present.
To create a new manifest explicitly at a new path without starting training:

```powershell
python .\bert_halueval\src\train_bert.py `
  --split-manifest .\data\splits\qa_split_seed42_experiment.json `
  --split-manifest-mode create `
  --prepare-splits-only
```

To reuse that manifest for training:

```powershell
python .\bert_halueval\src\train_bert.py `
  --split-manifest .\data\splits\qa_split_seed42_experiment.json `
  --split-manifest-mode reuse `
  --data .\data\qa_data.json `
  --model bert-base-uncased `
  --output-dir .\outputs\bert-halueval-qa `
  --epochs 3 `
  --batch-size 2
```

Use that same manifest path and split configuration for every model comparison.
An intentional replacement requires both `--split-manifest-mode create` and
`--replace-split-manifest`; normally create a different manifest path instead.
The dataset fingerprint ensures a manifest is not accidentally reused for
different or modified data. A limited smoke-test dataset is a different
dataset, so give it its own manifest.
Add `--prepare-splits-only` to the reuse command to validate a manifest without
training.

## Train BERT

Run this command from the project root:

```powershell
python .\bert_halueval\src\train_bert.py `
  --data .\data\qa_data.json `
  --model bert-base-uncased `
  --output-dir .\outputs\bert-halueval-qa `
  --split-manifest .\data\splits\qa_split_seed42.json `
  --split-manifest-mode auto `
  --epochs 3 `
  --batch-size 2 `
  --gradient-accumulation-steps 1 `
  --precision auto
```

If the existing environment is not activated, replace `python` with
`.\.venv\Scripts\python.exe`.

Quick smoke test using a separate 100-record split manifest and output folder:

```powershell
python .\bert_halueval\src\train_bert.py `
  --data .\data\qa_data.json `
  --limit 100 `
  --epochs 1 `
  --batch-size 2 `
  --split-manifest .\data\splits\qa_smoke_seed42.json `
  --split-manifest-mode create `
  --output-dir .\outputs\bert-smoke
```

The effective batch size is
`per-device batch size × gradient accumulation steps × number of devices`.
For example, batch size 2 and accumulation 4 on one GPU yields an effective
batch size of 8. The default accumulation is 1. On a 4 GB GPU, start with batch
size 2 and reduce it if CUDA runs out of memory.

Training selects the best checkpoint using validation hallucination-class F1.
The test set is evaluated only after that checkpoint is selected and does not
participate in checkpoint selection or parameter tuning. Non-finite training
or validation loss stops the run with an error.

### Input truncation and padding

The model input uses the same field format as before:

```text
Knowledge: {knowledge}
Question: {question}
Candidate answer: {answer}
```

Inputs that fit are tokenized as a whole. When an input exceeds `--max-length`,
the knowledge passage is truncated first; the question and candidate answer
are retained if they fit. If they do not fit together, the candidate answer is
prioritized and the question is truncated from its end. If the answer alone is
too long, it is truncated from its end. Model special tokens are included in
the length limit. Batches use dynamic padding to the longest sequence in that
batch rather than padding every example to the maximum. Truncation counts are
printed for each dataset split.

## Outputs and schemas

For `--output-dir .\outputs\bert-halueval-qa`, training writes:

- `model/`: the selected model and tokenizer.
- `checkpoints/`: Trainer checkpoints used for validation-based selection.
- `test_results.json`: stable JSON containing schema version, label mapping,
  accuracy, hallucination precision/recall/F1, macro-F1, hallucinated-class
  ROC-AUC, confusion matrix, full classification report, and experiment
  metadata. Undefined ROC-AUC is `null` with an explanatory reason.
- `test_predictions.csv`: one row per test example with
  `source_record_id`, `true_label`, `predicted_label`,
  `probability_not_hallucinated`, and `probability_hallucinated`.
- `experiment_metadata.json`: standalone metadata, including model checkpoint,
  seed, max sequence length, batch/accumulation/effective batch sizes, learning
  rate, epochs, precision, split-manifest path, training duration, peak GPU
  memory (`null` when unavailable), and test inference latency.

Inference latency is measured for one complete test-set inference pass after
model loading, including data-loader overhead, with CUDA synchronization before
and after the pass. The reported latency is average milliseconds per example.

## Interactive prediction

```powershell
python .\bert_halueval\src\predict.py `
  --model-dir .\outputs\bert-halueval-qa\model
```

Enter knowledge, question, and candidate answer at the prompts. The app uses
the same shared tokenizer truncation behavior as training, and uses CUDA when
available.

## Interpreting results and comparing models

- **Accuracy** is the fraction of correctly classified test examples.
- **Hallucination precision** is the fraction of predicted hallucinations that
  are labeled hallucinated.
- **Hallucination recall** is the fraction of labeled hallucinations detected.
- **Hallucination F1** is the harmonic mean of hallucination precision and
  recall. **Macro-F1** averages F1 across both classes.
- **ROC-AUC** measures ranking quality from the hallucinated-class probability;
  it is unavailable if the test labels contain only one class.

Compare models using the same dataset, split manifest, and held-out test set.
HaluEval labels are benchmark annotations, not a universal guarantee of truth.
High in-domain benchmark scores do not establish reliable performance on
arbitrary domain-specific inputs; collect and evaluate domain-representative
held-out labels before relying on a detector.

## Dependencies

Install the root `requirements.txt`. No additional evaluation or test framework
dependency is required; the focused unit tests use the Python standard-library
`unittest` runner.

## Citation

Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). *HaluEval: A
Large-Scale Hallucination Evaluation Benchmark for Large Language Models*.
https://arxiv.org/abs/2305.11747

Dataset repository: https://github.com/RUCAIBox/HaluEval
