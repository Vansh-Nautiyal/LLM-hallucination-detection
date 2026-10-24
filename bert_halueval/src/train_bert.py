"""Fine-tune BERT to classify HaluEval QA answers as factual or hallucinated.
The model receives knowledge + question + candidate answer, so it can judge the
candidate in context instead of learning from the answer alone.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)


LABELS = {0: "not_hallucinated", 1: "hallucinated"}


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read either standard JSON (list of objects) or JSONL (one object per line)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Put HaluEval's qa_data.json in the project's data/ folder, or pass --data PATH."
        )
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset file is empty: {path}")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Handles JSON objects wrapping records in a common collection key.
            for key in ("data", "records", "examples"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
            else:
                parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("JSON dataset must contain a list of records.")
        records = parsed
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON/JSONL at line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"JSONL line {line_no} is not a JSON object.")
            records.append(item)
    if not records:
        raise ValueError(f"No records found in {path}")
    if not all(isinstance(row, dict) for row in records):
        raise ValueError("Every dataset record must be a JSON object.")
    return records


def make_examples(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert every QA record into a factual and hallucinated answer example."""
    required = {"knowledge", "question", "right_answer", "hallucinated_answer"}
    missing = required - set().union(*(row.keys() for row in records))
    if missing:
        sample_keys = sorted(records[0].keys())
        raise ValueError(
            f"Missing required HaluEval fields: {sorted(missing)}. "
            f"First record contains: {sample_keys}"
        )

    examples: list[dict[str, Any]] = []
    skipped = 0
    for idx, row in enumerate(records):
        knowledge = str(row.get("knowledge", "") or "").strip()
        question = str(row.get("question", "") or "").strip()
        right = str(row.get("right_answer", "") or "").strip()
        hallucinated = str(row.get("hallucinated_answer", "") or "").strip()
        if not question or (not right and not hallucinated):
            skipped += 1
            continue
        # Use structured markers to make field roles explicit to the encoder.
        prefix = f"Knowledge: {knowledge}\nQuestion: {question}\nCandidate answer: "
        if right:
            examples.append({"text": prefix + right, "label": 0, "source_id": idx})
        if hallucinated:
            examples.append({"text": prefix + hallucinated, "label": 1, "source_id": idx})
    if skipped:
        print(f"Skipped {skipped} rows with empty question or both answers missing.")
    if not examples:
        raise ValueError("No usable examples after preprocessing.")
    return examples


class HaluEvalDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples = examples
        self.encodings = tokenizer(
            [item["text"] for item in examples],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx], dtype=torch.long)
                for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.examples[idx]["label"], dtype=torch.long)
        return item


def split_by_source(examples: list[dict[str, Any]], test_size: float,
                    val_size: float, seed: int):
    """Split by original QA record so paired right/hallucinated answers don't leak."""
    source_ids = sorted({e["source_id"] for e in examples})
    labels_by_source: dict[int, int] = {}
    for e in examples:
        # Original record is generally represented by both classes.
        labels_by_source.setdefault(e["source_id"], e["label"])
    train_val_ids, test_ids = train_test_split(
        source_ids, test_size=test_size, random_state=seed,
        stratify=[labels_by_source[i] for i in source_ids],
    )
    val_fraction_of_remaining = val_size / (1.0 - test_size)
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=val_fraction_of_remaining, random_state=seed,
        stratify=[labels_by_source[i] for i in train_val_ids],
    )
    train_set, val_set, test_set = set(train_ids), set(val_ids), set(test_ids)
    return (
        [e for e in examples if e["source_id"] in train_set],
        [e for e in examples if e["source_id"] in val_set],
        [e for e in examples if e["source_id"] in test_set],
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_hallucinated": precision_score(labels, predictions, pos_label=1, zero_division=0),
        "recall_hallucinated": recall_score(labels, predictions, pos_label=1, zero_division=0),
        "f1_hallucinated": f1_score(labels, predictions, pos_label=1, zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/qa_data.json"))
    parser.add_argument("--model", default="bert-base-uncased",
                        help="Hugging Face pretrained model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/bert-halueval-qa"))
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional number of source records for a quick smoke test (0 = all)")
    args = parser.parse_args()

    if not (0 < args.test_size < 1 and 0 < args.val_size < 1 and args.test_size + args.val_size < 1):
        raise ValueError("test-size and val-size must be >0 and their sum must be <1.")
    if args.max_length < 32:
        raise ValueError("max-length should be at least 32.")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        print(f"Training device: {torch.cuda.get_device_name(0)} "
              f"(CUDA {torch.version.cuda}; fp16 enabled)")
    else:
        print("Training device: CPU (CUDA unavailable; fp16 disabled)")

    records = read_records(args.data)
    if args.limit > 0:
        records = records[:args.limit]
    examples = make_examples(records)
    train_examples, val_examples, test_examples = split_by_source(
        examples, args.test_size, args.val_size, args.seed
    )

    print(f"Source records: {len(records):,}")
    print(f"Classification examples: {len(examples):,}")
    print(f"Train/validation/test examples: {len(train_examples):,} / "
          f"{len(val_examples):,} / {len(test_examples):,}")
    print("Label mapping: 0 = not hallucinated, 1 = hallucinated")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=LABELS,
        label2id={v: k for k, v in LABELS.items()},
    )

    train_dataset = HaluEvalDataset(train_examples, tokenizer, args.max_length)
    val_dataset = HaluEvalDataset(val_examples, tokenizer, args.max_length)
    test_dataset = HaluEvalDataset(test_examples, tokenizer, args.max_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_hallucinated",
        greater_is_better=True,
        logging_steps=100,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    trainer.train()

    test_result = trainer.predict(test_dataset)
    metrics = compute_metrics((test_result.predictions, test_result.label_ids))
    y_pred = np.argmax(test_result.predictions, axis=-1)
    y_true = test_result.label_ids
    report = classification_report(
        y_true, y_pred, labels=[0, 1],
        target_names=[LABELS[0], LABELS[1]], output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    final_model_dir = args.output_dir / "model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    results = {
        "model_checkpoint": args.model,
        "source_records": len(records),
        "classification_examples": len(examples),
        "split_sizes": {"train": len(train_examples), "validation": len(val_examples), "test": len(test_examples)},
        "label_mapping": {str(k): v for k, v in LABELS.items()},
        "test_metrics": metrics,
        "classification_report": report,
        "confusion_matrix_rows_true_cols_predicted": cm,
        "model_path": str(final_model_dir),
    }
    (args.output_dir / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nTest metrics:")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=[0, 1],
                                target_names=[LABELS[0], LABELS[1]], zero_division=0))
    print(f"Saved model to: {final_model_dir}")
    print(f"Saved test report to: {args.output_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
