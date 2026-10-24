"""Interactively classify answers and save a prediction report."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from common.evaluation import compute_metrics, generate_confusion_matrix
from common.preprocessing import LABELS, tokenize_example


def summarize(records: list[dict]) -> dict:
    labeled = [record for record in records if record["true_label"] is not None]
    if not labeled:
        return {
            "labeled_predictions": 0,
            "unlabeled_predictions": len(records),
            "accuracy": None,
            "precision_hallucinated": None,
            "recall_hallucinated": None,
            "f1_hallucinated": None,
            "macro_f1": None,
            "confusion_matrix_rows_true_cols_predicted": None,
        }

    true_labels = [record["true_label"] for record in labeled]
    predicted_labels = [record["predicted_label"] for record in labeled]
    probabilities = [
        [
            record["probability_not_hallucinated"],
            record["probability_hallucinated"],
        ]
        for record in labeled
    ]
    metrics = compute_metrics(true_labels, predicted_labels, probabilities)
    return {
        "labeled_predictions": len(labeled),
        "unlabeled_predictions": len(records) - len(labeled),
        **metrics,
        "confusion_matrix_rows_true_cols_predicted": generate_confusion_matrix(
            true_labels, predicted_labels
        ),
    }


def write_report(path: Path, records: list[dict]) -> None:
    report = {
        "created_at": datetime.now().astimezone().isoformat(),
        "prediction_count": len(records),
        "accuracy_metrics": summarize(records),
        "predictions": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively classify candidate answers as hallucinated or not."
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=REPOSITORY_ROOT / "outputs" / "bert-halueval-qa" / "model",
    )
    parser.add_argument("--report-path", type=Path, default=None,
                        help="JSON report path (defaults to a timestamped file in outputs).")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    if args.max_length < 1:
        raise ValueError("max-length must be greater than zero.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Loaded model on {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}.")
    print("Enter 'exit' at any prompt to finish and save the report.")
    print("After each prediction, enter y/n for its known label, or skip if unknown.\n")

    records: list[dict] = []
    should_exit = False
    try:
        while not should_exit:
            knowledge = input("Knowledge / relevant facts: ").strip()
            if knowledge.casefold() == "exit":
                break
            question = input("Question: ").strip()
            if question.casefold() == "exit":
                break
            answer = input("Candidate answer to check: ").strip()
            if answer.casefold() == "exit":
                break
            if not knowledge or not question or not answer:
                print("Knowledge, question, and candidate answer must not be empty.\n")
                continue

            encoding = tokenize_example(
                tokenizer, knowledge, question, answer, args.max_length
            )
            inputs = {
                key: torch.tensor([value], dtype=torch.long, device=device)
                for key, value in encoding.items()
            }
            with torch.inference_mode():
                probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]
            predicted_label = int(torch.argmax(probabilities).item())
            predicted_name = LABELS[predicted_label]

            print(f"\nPrediction: {predicted_name.upper().replace('_', ' ')}")
            print(f"P(not hallucinated): {probabilities[0].item():.4f}")
            print(f"P(hallucinated):     {probabilities[1].item():.4f}")
            record = {
                "knowledge": knowledge,
                "question": question,
                "answer": answer,
                "predicted_label": predicted_label,
                "prediction": predicted_name,
                "probability_not_hallucinated": probabilities[0].item(),
                "probability_hallucinated": probabilities[1].item(),
                "true_label": None,
            }
            records.append(record)

            while True:
                truth = input(
                    "Was the answer actually hallucinated? [y/n/skip/exit]: "
                ).strip().casefold()
                if truth in {"y", "yes"}:
                    record["true_label"] = 1
                    break
                if truth in {"n", "no"}:
                    record["true_label"] = 0
                    break
                if truth == "skip":
                    break
                if truth == "exit":
                    should_exit = True
                    break
                print("Enter y, n, skip, or exit.")
            print()
    except (KeyboardInterrupt, EOFError):
        print("\nInput ended; saving the report.")

    report_path = args.report_path
    if report_path is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        report_path = Path("outputs") / f"prediction-report-{timestamp}.json"
    write_report(report_path, records)

    metrics = summarize(records)
    print(f"Saved {len(records)} prediction(s) to: {report_path}")
    if metrics["labeled_predictions"]:
        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Hallucination precision: {metrics['precision_hallucinated']:.4f} | "
            f"recall: {metrics['recall_hallucinated']:.4f} | "
            f"F1: {metrics['f1_hallucinated']:.4f}"
        )
    else:
        print("No ground-truth labels were recorded; accuracy metrics are unavailable.")
    print("Prediction scores are model estimates, not guarantees of factual correctness.")


if __name__ == "__main__":
    main()
