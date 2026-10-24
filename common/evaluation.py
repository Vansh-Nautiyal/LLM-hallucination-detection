"""Model-independent evaluation metrics, prediction export, and measurements."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from common.preprocessing import LABELS


def _validated_arrays(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    probabilities: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    true_values = np.asarray(true_labels, dtype=int)
    predicted_values = np.asarray(predicted_labels, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    if true_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("True and predicted labels must be one-dimensional.")
    if len(true_values) == 0 or len(true_values) != len(predicted_values):
        raise ValueError("True and predicted labels must have equal nonzero lengths.")
    if not set(true_values).issubset(LABELS) or not set(predicted_values).issubset(LABELS):
        raise ValueError("Labels must use the established class IDs 0 and 1.")
    if probability_values.ndim == 2:
        if probability_values.shape != (len(true_values), 2):
            raise ValueError("Class probabilities must have shape (n_examples, 2).")
        hallucinated_probability = probability_values[:, 1]
    elif probability_values.ndim == 1 and len(probability_values) == len(true_values):
        hallucinated_probability = probability_values
    else:
        raise ValueError("Probabilities must be a class-1 vector or an (n, 2) array.")
    if not np.isfinite(hallucinated_probability).all():
        raise ValueError("Probabilities must contain only finite values.")
    return true_values, predicted_values, hallucinated_probability


def compute_metrics(
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    probabilities: Sequence[Sequence[float]] | Sequence[float] | np.ndarray,
) -> dict[str, Any]:
    """Compute stable binary-classification metrics; undefined ROC-AUC is null."""
    true_values, predicted_values, hallucinated_probability = _validated_arrays(
        true_labels, predicted_labels, probabilities
    )
    roc_auc = None
    roc_auc_unavailable_reason = None
    if len(np.unique(true_values)) < 2:
        roc_auc_unavailable_reason = "ROC-AUC requires both true classes."
    else:
        roc_auc = float(roc_auc_score(true_values, hallucinated_probability))
    return {
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "precision_hallucinated": float(
            precision_score(true_values, predicted_values, pos_label=1, zero_division=0)
        ),
        "recall_hallucinated": float(
            recall_score(true_values, predicted_values, pos_label=1, zero_division=0)
        ),
        "f1_hallucinated": float(
            f1_score(true_values, predicted_values, pos_label=1, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(true_values, predicted_values, average="macro", zero_division=0)
        ),
        "roc_auc_hallucinated": roc_auc,
        "roc_auc_unavailable_reason": roc_auc_unavailable_reason,
    }


def generate_classification_report(
    true_labels: Sequence[int], predicted_labels: Sequence[int]
) -> dict[str, Any]:
    """Return a JSON-compatible per-class classification report."""
    return classification_report(
        true_labels,
        predicted_labels,
        labels=[0, 1],
        target_names=[LABELS[0], LABELS[1]],
        output_dict=True,
        zero_division=0,
    )


def generate_confusion_matrix(
    true_labels: Sequence[int], predicted_labels: Sequence[int]
) -> list[list[int]]:
    """Return a fixed-order confusion matrix (true rows, predicted columns)."""
    return confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).tolist()


def save_test_predictions(
    path: str | Path,
    examples: Sequence[dict[str, Any]],
    true_labels: Sequence[int],
    predicted_labels: Sequence[int],
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> None:
    """Export one aligned row per test example with both class probabilities."""
    true_values = np.asarray(true_labels, dtype=int)
    predicted_values = np.asarray(predicted_labels, dtype=int)
    probability_values = np.asarray(probabilities, dtype=float)
    count = len(examples)
    if (count == 0 or len(true_values) != count or len(predicted_values) != count
            or probability_values.shape != (count, 2)):
        raise ValueError(
            "Test examples, labels, predictions, and probabilities must have "
            "matching nonzero row counts."
        )
    _validated_arrays(true_values, predicted_values, probability_values)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "source_record_id",
                "true_label",
                "predicted_label",
                "probability_not_hallucinated",
                "probability_hallucinated",
            ),
        )
        writer.writeheader()
        for example, true_label, predicted_label, class_probabilities in zip(
            examples, true_values, predicted_values, probability_values
        ):
            writer.writerow(
                {
                    "source_record_id": example["source_id"],
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                    "probability_not_hallucinated": float(class_probabilities[0]),
                    "probability_hallucinated": float(class_probabilities[1]),
                }
            )
    with output.open("r", encoding="utf-8", newline="") as handle:
        written_rows = sum(1 for _ in csv.DictReader(handle))
    if written_rows != count:
        raise RuntimeError(
            f"Prediction export wrote {written_rows} rows for {count} test examples."
        )


def save_evaluation_results(
    path: str | Path,
    *,
    metrics: dict[str, Any],
    report: dict[str, Any],
    matrix: list[list[int]],
    metadata: dict[str, Any],
) -> None:
    """Save a consistent JSON evaluation result and a standalone metadata file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "label_mapping": {str(key): value for key, value in LABELS.items()},
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix_rows_true_cols_predicted": matrix,
        "experiment_metadata": metadata,
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    metadata_path = output.parent / "experiment_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )


def measure_inference_latency(
    inference: Callable[[], Any],
    *,
    number_of_examples: int,
    device: torch.device | str,
) -> tuple[Any, dict[str, float | str]]:
    """Time one complete inference call after model loading, reporting per-example ms.

    The call is synchronized before and after on CUDA so asynchronous kernels are
    included. The result covers the callable's full inference pass, including its
    data-loader overhead, and does not include model-loading time.
    """
    if number_of_examples < 1:
        raise ValueError("number_of_examples must be greater than zero.")
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = inference()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return output, {
        "measurement": "one complete inference pass; model loading excluded",
        "latency_scope": "milliseconds per example, averaged across test examples",
        "elapsed_seconds": elapsed,
        "milliseconds_per_example": elapsed * 1000.0 / number_of_examples,
    }


def peak_gpu_memory_mb(device: torch.device | str) -> float | None:
    """Return peak allocated CUDA memory in MiB, or None when unavailable."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
