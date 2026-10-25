"""Fine-tune DeBERTa-v3 to detect hallucinated HaluEval QA answers."""

from __future__ import annotations

import argparse
import inspect
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Resolve the project root so this script can import shared project code.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.evaluation import (
    compute_metrics,
    generate_classification_report,
    generate_confusion_matrix,
    measure_inference_latency,
    peak_gpu_memory_mb,
    save_evaluation_results,
    save_test_predictions,
)
from common.preprocessing import (
    LABELS,
    create_or_load_splits,
    make_examples,
    read_records,
    tokenize_example_with_stats,
)


DEFAULT_MODEL = "microsoft/deberta-v3-base"


class HaluEvalDataset(Dataset):
    """Tokenize QA examples and record truncation statistics."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples = examples
        self.encodings: list[dict[str, list[int]]] = []
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
    """Stop training if training or validation loss becomes invalid."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not np.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite training loss at step {state.global_step}: {loss}"
            )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        loss = (metrics or {}).get("eval_loss")
        if loss is not None and not np.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite validation loss at step {state.global_step}: {loss}"
            )


def promote_run_outputs(output_dir: Path, staging_dir: Path) -> None:
    """Replace the final model and reports only after all staged files exist."""
    required_files = (
        "test_predictions.csv",
        "test_results.json",
        "experiment_metadata.json",
    )
    staged_model = staging_dir / "model"
    if not staged_model.is_dir() or not any(staged_model.iterdir()):
        raise RuntimeError(f"Staged model is missing or empty: {staged_model}")
    for filename in required_files:
        if not (staging_dir / filename).is_file():
            raise RuntimeError(f"Required staged output is missing: {filename}")

    backup_dir = staging_dir / "previous-output"
    backup_dir.mkdir()
    targets = [(staged_model, output_dir / "model")]
    targets.extend(
        (staging_dir / filename, output_dir / filename)
        for filename in required_files
    )
    backed_up: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for staged_path, destination in targets:
            if destination.exists():
                if destination.is_dir() != staged_path.is_dir():
                    raise RuntimeError(
                        f"Cannot safely replace output with incompatible type: {destination}"
                    )
                backup_path = backup_dir / destination.name
                os.replace(destination, backup_path)
                backed_up.append((backup_path, destination))
            os.replace(staged_path, destination)
            promoted.append(destination)
    except BaseException:
        for destination in reversed(promoted):
            if destination.is_dir():
                shutil.rmtree(destination)
            elif destination.exists():
                destination.unlink()
        for backup_path, destination in reversed(backed_up):
            os.replace(backup_path, destination)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "qa_data.json",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "deberta-halueval-qa",
    )
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit source records for a smoke test; 0 uses all records.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "splits" / "qa_split_seed42.json",
    )
    parser.add_argument(
    "--split-manifest-mode",
    choices=("auto", "create", "reuse"),
    default="reuse",
    help="Reuse the existing BERT split manifest; never create a new one by default.",
    )
    parser.add_argument(
        "--replace-split-manifest",
        action="store_true",
        help="Allow --split-manifest-mode create to replace an existing manifest.",
    )
    parser.add_argument(
        "--prepare-splits-only",
        action="store_true",
        help="Validate or create splits without loading a model.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "no", "fp16", "bf16"),
        default="auto",
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("Batch size and gradient accumulation must be positive.")
    if args.max_length < 1:
        parser.error("--max-length must be greater than zero.")
    if args.limit < 0:
        parser.error("--limit cannot be negative.")
    if args.replace_split_manifest and args.split_manifest_mode != "create":
        parser.error(
            "--replace-split-manifest requires --split-manifest-mode create."
        )
    if args.limit > 0 and args.split_manifest_mode == "auto":
        parser.error(
            "--limit changes the dataset fingerprint. For a smoke test, use a "
            "separate --split-manifest path and --split-manifest-mode create "
            "on the first run, then reuse that same smoke manifest."
        )
    if not (
        0 < args.test_size < 1
        and 0 < args.val_size < 1
        and args.test_size + args.val_size < 1
    ):
        parser.error("Split proportions must be positive and sum to less than 1.")
    return args


def get_precision_flags(precision: str) -> tuple[bool, bool]:
    """Return (use_fp16, use_bf16), validating GPU support."""

    cuda_available = torch.cuda.is_available()

    if precision == "auto":
        return cuda_available, False
    if precision == "no":
        return False, False
    if not cuda_available:
        raise RuntimeError(
            f"--precision {precision} requires a CUDA-enabled GPU."
        )
    if precision == "fp16":
        return True, False
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 was requested, but the current GPU does not support it."
        )

    return False, True


def trainer_compute_metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    probabilities = torch.softmax(
        torch.as_tensor(logits), dim=-1
    ).cpu().numpy()
    predictions = np.argmax(probabilities, axis=-1)

    metrics = compute_metrics(labels, predictions, probabilities)

    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float, np.number))
        and value is not None
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(args.seed)
        print(f"Training device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Training device: CPU")

    # Load and preprocess the dataset using shared project code.
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

    print(f"Model checkpoint: {args.model}")
    print(f"Split manifest: {args.split_manifest}")
    print(f"Source records loaded: {len(records):,}")
    print(f"Classification examples: {len(examples):,}")

    for name, partition in (
        ("train", train_examples),
        ("validation", val_examples),
        ("test", test_examples),
    ):
        labels = [item["label"] for item in partition]
        source_count = len({item["source_id"] for item in partition})
        print(
            f"{name}: {len(partition):,} examples, "
            f"{source_count:,} source records, "
            f"class 0={labels.count(0)}, class 1={labels.count(1)}"
        )

    print("Labels: 0 = not hallucinated; 1 = hallucinated")

    if args.prepare_splits_only:
        print("Split preparation complete; model loading skipped.")
        return

    # Load DeBERTa-v3 tokenizer and classification model.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label=LABELS,
        label2id={value: key for key, value in LABELS.items()},
    )

    train_dataset = HaluEvalDataset(
        train_examples, tokenizer, args.max_length
    )
    val_dataset = HaluEvalDataset(
        val_examples, tokenizer, args.max_length
    )
    test_dataset = HaluEvalDataset(
        test_examples, tokenizer, args.max_length
    )

    for name, dataset in (
        ("train", train_dataset),
        ("validation", val_dataset),
        ("test", test_dataset),
    ):
        print(
            f"{name} truncation: "
            f"knowledge={dataset.knowledge_truncated}, "
            f"question/answer={dataset.question_or_answer_truncated}"
        )

    use_fp16, use_bf16 = get_precision_flags(args.precision)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Support both older and newer Transformers versions.
    training_parameters = inspect.signature(
        TrainingArguments
    ).parameters
    evaluation_argument = (
        "eval_strategy"
        if "eval_strategy" in training_parameters
        else "evaluation_strategy"
    )

    training_config = {
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
    training_config[evaluation_argument] = "epoch"

    training_args = TrainingArguments(**training_config)

    trainer_config = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": trainer_compute_metrics,
        "callbacks": [FailOnNonFiniteLoss()],
    }

    trainer_parameters = inspect.signature(Trainer).parameters
    if "processing_class" in trainer_parameters:
        trainer_config["processing_class"] = tokenizer
    else:
        trainer_config["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".run-staging-", dir=args.output_dir)
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    training_start = time.perf_counter()
    trainer.train()
    training_seconds = time.perf_counter() - training_start
    peak_memory = peak_gpu_memory_mb(device)

    # Save the best model and tokenizer independently of BERT outputs.
    final_model_dir = args.output_dir / "model"
    staged_model_dir = staging_dir / "model"
    trainer.save_model(str(staged_model_dir))
    tokenizer.save_pretrained(str(staged_model_dir))

    test_result, latency = measure_inference_latency(
        lambda: trainer.predict(test_dataset),
        number_of_examples=len(test_dataset),
        device=device,
    )

    true_labels = np.asarray(test_result.label_ids, dtype=int)
    probabilities = torch.softmax(
        torch.as_tensor(test_result.predictions), dim=-1
    ).cpu().numpy()
    predicted_labels = np.argmax(probabilities, axis=-1)

    metrics = compute_metrics(
        true_labels, predicted_labels, probabilities
    )
    report = generate_classification_report(
        true_labels, predicted_labels
    )
    matrix = generate_confusion_matrix(
        true_labels, predicted_labels
    )

    save_test_predictions(
        staging_dir / "test_predictions.csv",
        test_examples,
        true_labels,
        predicted_labels,
        probabilities,
    )

    metadata = {
        "model_checkpoint": args.model,
        "random_seed": args.seed,
        "max_sequence_length": args.max_length,
        "per_device_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "number_of_devices": max(1, training_args.world_size),
        "effective_batch_size": (
            args.batch_size
            * args.gradient_accumulation_steps
            * max(1, training_args.world_size)
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
        "training_duration_seconds": training_seconds,
        "peak_gpu_memory_mib": peak_memory,
        "inference_latency": latency,
        "best_checkpoint": trainer.state.best_model_checkpoint,
    }

    save_evaluation_results(
        staging_dir / "test_results.json",
        metrics=metrics,
        report=report,
        matrix=matrix,
        metadata=metadata,
    )

    promote_run_outputs(args.output_dir, staging_dir)
    shutil.rmtree(staging_dir)
    checkpoints_dir = args.output_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        shutil.rmtree(checkpoints_dir)

    print("\nFinal test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"Model saved to: {final_model_dir}")
    print(f"Predictions saved to: {args.output_dir / 'test_predictions.csv'}")
    print(f"Metrics saved to: {args.output_dir / 'test_results.json'}")
    print(f"Training duration: {training_seconds:.2f} seconds")
    print(
        "Inference latency: "
        f"{latency['milliseconds_per_example']:.3f} ms/example"
    )


if __name__ == "__main__":
    main()
