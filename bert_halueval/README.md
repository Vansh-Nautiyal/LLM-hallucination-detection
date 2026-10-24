# BERT-based Hallucination Detection with HaluEval QA

Fine-tunes `bert-base-uncased` as a binary classifier on HaluEval's QA task.

## 1. Dataset placement

Use the repository's QA dataset and keep your current directory layout:

```text
LLM-Hallucination-Detection/
├── data/
│   ├── qa_data.json        # HaluEval QA file (JSON or JSONL accepted)
│   ├── dialogue_data.json
│   ├── general_data.json
│   └── summarization_data.json
├── bert_halueval/
│   ├── README.md
│   └── src/
│       ├── train_bert.py
│       └── predict.py
├── requirements.txt
└── .venv/
```

The HaluEval repository currently documents the file as `data/qa_data.json` (JSON) with `knowledge`, `question`, `right_answer`, and `hallucinated_answer` fields. Your screenshot's `qa_data.json` filename is fine; no conversion to `.jsonl` is needed. If your local file really is named `qa_data.jsonl`, pass it via `--data data/qa_data.jsonl`.

## 2. Environment setup

Open a terminal at the project root, activate your virtual environment, then run:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For this project's Windows NVIDIA GPU setup, install the CUDA-enabled PyTorch build after the other requirements:

```powershell
python -m pip install --upgrade torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

The training script automatically uses CUDA and mixed precision when `torch.cuda.is_available()` is true; it prints the selected device before training. If adapting this setup to another GPU, choose a compatible PyTorch build from the official [PyTorch installation selector](https://pytorch.org/get-started/locally/). CPU training remains available but may be slow.

## 3. Quick smoke test

First run on 100 source records to verify the data, environment, tokenizer, and training loop:

```bash
python .\bert_halueval\src\train_bert.py --data .\data\qa_data.json --limit 100 --epochs 1 --batch-size 2
```

If the file uses `.jsonl`, replace the path accordingly. The model downloads from Hugging Face the first time.

## 4. Train on the full QA dataset

```bash
python .\bert_halueval\src\train_bert.py --data .\data\qa_data.json --epochs 3 --batch-size 2
```

The batch size of 2 is a safer starting point for GPUs with 4 GB of memory. If your GPU has more memory, increase it gradually. To reduce sequence truncation, test `--max-length 384` if GPU memory allows. Defaults are `bert-base-uncased`, max length 256, learning rate 2e-5, and 3 epochs.

Artifacts go to:

```text
outputs/bert-halueval-qa/
├── checkpoints/
├── model/             # best fine-tuned model and tokenizer
└── test_results.json  # held-out metrics and confusion matrix
```

The split is made by original source record, so the correct and hallucinated answers associated with one question do not cross between train, validation, and test sets. The test set is only used for final evaluation.

## 5. Interactive prediction and report

```bash
python .\bert_halueval\src\predict.py --model-dir .\outputs\bert-halueval-qa\model
```

Enter the relevant knowledge/facts, question, and candidate answer when prompted. After each result, enter `y` or `n` to record its known ground-truth label, or `skip` if the truth is unknown. Type `exit` at any prompt to finish. A JSON report containing each prediction and accuracy metrics for labeled predictions is saved under `outputs/`; pass `--report-path PATH` to choose another file. The model uses CUDA automatically when available.

## 6. What the labels mean

- `0`: `right_answer`, a not-hallucinated / reference-grounded answer.
- `1`: `hallucinated_answer`, a hallucinated answer produced for the benchmark.

For each record, the training script creates two examples. Each model input includes the knowledge, question, and candidate answer. This makes the task a supervised, context-aware answer classification problem. At inference time, provide the relevant knowledge as well as the question and answer.

## 7. Evaluation and limitations

Review `test_results.json` and focus on hallucination-class precision, recall and F1, as well as macro-F1 and the confusion matrix. Accuracy alone may hide which class is harder to detect. The generated `right_answer` and `hallucinated_answer` labels are benchmark labels, not a universal guarantee that every answer is objectively true or false. Results may not generalize to other domains or model styles without further evaluation.

## Citation

Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). *HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models*. https://arxiv.org/abs/2305.11747

Dataset repository: https://github.com/RUCAIBox/HaluEval
