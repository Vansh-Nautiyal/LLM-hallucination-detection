"""Fine-tune BERT to classify HaluEval QA answers as factual or hallucinated."""
from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.evaluation import (compute_metrics, generate_classification_report,
                               generate_confusion_matrix,
                               measure_inference_latency, peak_gpu_memory_mb,
                               save_evaluation_results, save_test_predictions)
from common.preprocessing import (LABELS, create_or_load_splits, make_examples,
                                  read_records, tokenize_example_with_stats)


class HaluEvalDataset(Dataset):
    """Tokenize examples with knowledge-first truncation."""

    def __init__(self, examples: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples = examples
        self.encodings = []
        self.knowledge_truncated = 0
        self.question_or_answer_truncated = 0
        for example in examples:
            encoding, stats = tokenize_example_with_stats(
                tokenizer,
                example["knowledge"],
                example["question"],
                example["answer"],
                max_length,
            )
            self.encodings.append(encoding)
            self.knowledge_truncated += int(stats["knowledge_truncated"])
            self.question_or_answer_truncated += int(
                stats["question_or_answer_truncated"]
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in self.encodings[index].items()
        }
        item["labels"] = torch.tensor(
            self.examples[index]["label"], dtype=torch.long
        )
        return item


class FailOnNonFiniteLoss(TrainerCallback):
    """Stop training immediately if a logged training or validation loss is invalid."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not np.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at global step {state.global_step}: {loss}"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is not None and not np.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss at global step {state.global_step}: {loss}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=REPOSITORY_ROOT / "data" / "qa_data.json"
    )
    parser.add_argument("--model", default="bert-base-uncased",
                        help="Hugging Face pretrained model checkpoint")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY_ROOT / "outputs" / "bert-halueval-qa",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Optional source-record limit for a smoke test (0 means all records)",
    )
    parser.add_argument(
        "--split-manifest", type=Path,
        default=REPOSITORY_ROOT / "data" / "splits" / "qa_split_seed42.json",
    )
    parser.add_argument(
        "--split-manifest-mode",
        choices=("auto", "create", "reuse"),
        default="auto",
        help="auto creates a missing manifest and reuses an existing compatible one",
    )
    parser.add_argument(
        "--replace-split-manifest", action="store_true",
        help="Allow --split-manifest-mode create to replace an existing manifest",
    )
    parser.add_argument(
        "--prepare-splits-only", action="store_true",
        help="Create or validate the split manifest without loading a model or training",
    )
    parser.add_argument(
        "--precision", choices=("auto", "no", "fp16", "bf16"), default="auto",
        help="auto uses FP16 on CUDA and full precision on CPU",
    )
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("--batch-size and --gradient-accumulation-steps must be positive.")
    if args.max_length < 1:
        parser.error("--max-length must be greater than zero.")
    if args.limit < 0:
        parser.error("--limit must not be negative.")
    if args.replace_split_manifest and args.split_manifest_mode != "create":
        parser.error("--replace-split-manifest requires --split-manifest-mode create.")
    return args


def _precision_flags(precision: str) -> tuple[bool, bool]:
    cuda_available = torch.cuda.is_available()
    if precision == "auto":
        return cuda_available, False
    if precision == "no":
        return False, False
    if not cuda_available:
        raise RuntimeError(f"--precision {precision} requires a CUDA-enabled GPU.")
    if precision == "fp16":
        return True, False
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("--precision bf16 was requested but this GPU does not support it.")
    return False, True


def _compute_trainer_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    probabilities = torch.softmax(torch.as_tensor(logits), dim=-1).cpu().numpy()
    predictions = np.argmax(probabilities, axis=-1)
    metrics = compute_metrics(labels, predictions, probabilities)
    return {
        key: value for key, value in metrics.items()
        if isinstance(value, (float, int)) and value is not None
    }


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        print(f"Training device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Training device: CPU")

    records = read_records(args.data)
    if args.limit:
        records = records[:args.limit]
    examples = make_examples(records)
    train_examples, val_examples, test_examples = create_or_load_splits(
        examples,
        records,
        args.split_manifest,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        mode=args.split_manifest_mode,
        replace=args.replace_split_manifest,
    )
    print(f"Split manifest: {args.split_manifest}")
    print(
        f"Source records: {len(records):,}; usable source records: "
        f"{len(examples) // 2:,}; classification examples: {len(examples):,}"
    )
    for name, partition in (
        ("train", train_examples),
        ("validation", val_examples),
        ("test", test_examples),
    ):
        print(
            f"{name}: {len(partition):,} examples from "
            f"{len({item['source_id'] for item in partition}):,} source records"
        )
    print("Label mapping: 0 = not hallucinated, 1 = hallucinated")
    if args.prepare_splits_only:
        print("Split preparation complete; model loading and training were skipped.")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=LABELS,
        label2id={value: key for key, value in LABELS.items()},
    )
    train_dataset = HaluEvalDataset(train_examples, tokenizer, args.max_length)
    val_dataset = HaluEvalDataset(val_examples, tokenizer, args.max_length)
    test_dataset = HaluEvalDataset(test_examples, tokenizer, args.max_length)
    for name, dataset in (
        ("train", train_dataset),
        ("validation", val_dataset),
        ("test", test_dataset),
    ):
        print(
            f"{name} truncation: knowledge={dataset.knowledge_truncated}, "
            f"question/answer={dataset.question_or_answer_truncated}"
        )

    use_fp16, use_bf16 = _precision_flags(args.precision)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_argument_parameters = inspect.signature(TrainingArguments).parameters
    eval_strategy_argument = (
        "eval_strategy"
        if "eval_strategy" in training_argument_parameters
        else "evaluation_strategy"
    )
    training_arguments = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_hallucinated",
        "greater_is_better": True,
        "logging_steps": 100,
        "save_total_limit": 2,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "logging_nan_inf_filter": False,
        "report_to": "none",
        "seed": args.seed,
        "data_seed": args.seed,
    }
    training_arguments[eval_strategy_argument] = "epoch"
    training_args = TrainingArguments(**training_arguments)
    trainer_options = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": _compute_trainer_metrics,
        "callbacks": [FailOnNonFiniteLoss()],
    }
    trainer_parameters = inspect.signature(Trainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_options["processing_class"] = tokenizer
    else:
        trainer_options["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_options)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    trainer.train()
    training_duration = time.perf_counter() - training_started
    peak_memory = peak_gpu_memory_mb(device)

    final_model_dir = args.output_dir / "model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    test_result, inference_latency = measure_inference_latency(
        lambda: trainer.predict(test_dataset),
        number_of_examples=len(test_dataset),
        device=device,
    )
    true_labels = np.asarray(test_result.label_ids, dtype=int)
    probabilities = torch.softmax(
        torch.as_tensor(test_result.predictions), dim=-1
    ).cpu().numpy()
    predicted_labels = np.argmax(probabilities, axis=-1)
    metrics = compute_metrics(true_labels, predicted_labels, probabilities)
    report = generate_classification_report(true_labels, predicted_labels)
    matrix = generate_confusion_matrix(true_labels, predicted_labels)
    save_test_predictions(
        args.output_dir / "test_predictions.csv",
        test_examples,
        true_labels,
        predicted_labels,
        probabilities,
    )

    world_size = max(1, training_args.world_size)
    metadata = {
        "model_checkpoint": args.model,
        "random_seed": args.seed,
        "max_sequence_length": args.max_length,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "number_of_devices": world_size,
        "effective_batch_size": (
            args.batch_size * args.gradient_accumulation_steps * world_size
        ),
        "learning_rate": args.learning_rate,
        "number_of_epochs": args.epochs,
        "precision": (
            "fp16" if use_fp16 else "bf16" if use_bf16 else "full"
        ),
        "split_manifest_path": str(args.split_manifest.resolve()),
        "source_record_count": len(records),
        "usable_source_record_count": len(examples) // 2,
        "example_counts": {
            "train": len(train_examples),
            "validation": len(val_examples),
            "test": len(test_examples),
        },
        "training_duration_seconds": training_duration,
        "peak_gpu_memory_mib": peak_memory,
        "inference_latency": inference_latency,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }
    save_evaluation_results(
        args.output_dir / "test_results.json",
        metrics=metrics,
        report=report,
        matrix=matrix,
        metadata=metadata,
    )

    print("\nFinal test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"Saved best model and tokenizer to: {final_model_dir}")
    print(f"Saved test predictions to: {args.output_dir / 'test_predictions.csv'}")
    print(f"Saved evaluation results to: {args.output_dir / 'test_results.json'}")
    print(
        f"Training duration: {training_duration:.2f}s; "
        f"inference latency: {inference_latency['milliseconds_per_example']:.3f} ms/example"
    )


if __name__ == "__main__":
    main()
