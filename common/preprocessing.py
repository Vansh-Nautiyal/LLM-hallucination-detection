"""Dataset preparation, source-level splitting, and tokenizer-safe truncation."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from sklearn.model_selection import train_test_split


LABELS = {0: "not_hallucinated", 1: "hallucinated"}
REQUIRED_FIELDS = ("knowledge", "question", "right_answer", "hallucinated_answer")
MANIFEST_VERSION = 1
_WRAPPER_KEYS = ("data", "records", "examples")


def read_records(data_path: str | Path) -> list[dict[str, Any]]:
    """Load HaluEval records from JSON, wrapped JSON, or JSONL."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Dataset file is empty: {path}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON/JSONL in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {line_number} must contain an object.")
            parsed.append(row)

    if isinstance(parsed, dict):
        for key in _WRAPPER_KEYS:
            if isinstance(parsed.get(key), list):
                parsed = parsed[key]
                break
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"Dataset in {path} must contain a list of objects.")
    if not parsed:
        raise ValueError(f"No records found in {path}.")
    if any(not isinstance(row, dict) for row in parsed):
        raise ValueError("Every dataset record must be a JSON object.")

    missing = set(REQUIRED_FIELDS) - set().union(*(row.keys() for row in parsed))
    if missing:
        raise ValueError(
            f"Dataset is missing required HaluEval field(s): {sorted(missing)}."
        )
    return parsed


def make_examples(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create paired labeled examples and report incomplete source records."""
    if not records:
        raise ValueError("Cannot create examples from an empty dataset.")
    if any(not isinstance(row, dict) for row in records):
        raise ValueError("Every dataset record must be a JSON object.")

    missing = set(REQUIRED_FIELDS) - set().union(*(row.keys() for row in records))
    if missing:
        raise ValueError(
            f"Dataset is missing required HaluEval field(s): {sorted(missing)}."
        )

    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for source_id, row in enumerate(records):
        values = {
            field: str(row.get(field, "") or "").strip()
            for field in REQUIRED_FIELDS
        }
        missing_values = [field for field, value in values.items() if not value]
        if missing_values:
            skipped[", ".join(missing_values)] += 1
            continue

        knowledge = values["knowledge"]
        question = values["question"]
        prefix = f"Knowledge: {knowledge}\nQuestion: {question}\nCandidate answer: "
        examples.extend(
            (
                {
                    "text": prefix + values["right_answer"],
                    "knowledge": knowledge,
                    "question": question,
                    "answer": values["right_answer"],
                    "label": 0,
                    "source_id": source_id,
                },
                {
                    "text": prefix + values["hallucinated_answer"],
                    "knowledge": knowledge,
                    "question": question,
                    "answer": values["hallucinated_answer"],
                    "label": 1,
                    "source_id": source_id,
                },
            )
        )

    if skipped:
        counts = "; ".join(
            f"{count} missing {reason}" for reason, count in sorted(skipped.items())
        )
        print(f"Skipped {sum(skipped.values())} incomplete source record(s): {counts}.")
    if not examples:
        raise ValueError("No complete, usable source records found.")
    return examples


def validate_examples(examples: Sequence[dict[str, Any]]) -> list[int]:
    """Validate paired source labels and return the sorted source IDs."""
    if not examples:
        raise ValueError("Cannot split an empty set of examples.")
    labels_by_source: dict[int, list[int]] = defaultdict(list)
    for example in examples:
        source_id = example.get("source_id")
        label = example.get("label")
        if not isinstance(source_id, int) or isinstance(source_id, bool):
            raise ValueError(f"Invalid source-record ID: {source_id!r}.")
        if not isinstance(label, int) or isinstance(label, bool) or label not in LABELS:
            raise ValueError(f"Invalid label {label!r} for source {source_id}.")
        labels_by_source[source_id].append(label)

    for source_id, labels in labels_by_source.items():
        if sorted(labels) != [0, 1]:
            raise ValueError(
                f"Source {source_id} must have exactly one example for each label; "
                f"found {labels}."
            )
    return sorted(labels_by_source)


def _validate_proportions(test_size: float, val_size: float) -> None:
    if not (0 < test_size < 1 and 0 < val_size < 1
            and test_size + val_size < 1):
        raise ValueError(
            "test_size and val_size must be greater than 0 and sum to less than 1."
        )


def _validate_partition_ids(
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    expected_ids: Sequence[int],
) -> None:
    partitions = [list(train_ids), list(val_ids), list(test_ids)]
    names = ("train", "validation", "test")
    for name, ids in zip(names, partitions):
        if len(ids) != len(set(ids)):
            raise ValueError(f"The {name} source IDs contain duplicates.")
    train_set, val_set, test_set = map(set, partitions)
    if not train_set.isdisjoint(val_set):
        raise ValueError("Train and validation source IDs overlap.")
    if not train_set.isdisjoint(test_set):
        raise ValueError("Train and test source IDs overlap.")
    if not val_set.isdisjoint(test_set):
        raise ValueError("Validation and test source IDs overlap.")
    actual = train_set | val_set | test_set
    expected = set(expected_ids)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"Split source IDs do not match the dataset (missing={missing[:10]}, "
            f"unexpected={unexpected[:10]})."
        )


def split_by_source(
    examples: Sequence[dict[str, Any]],
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split complete sources, keeping paired answers in one partition."""
    _validate_proportions(test_size, val_size)
    source_ids = validate_examples(examples)
    if len(source_ids) < 3:
        raise ValueError("At least three complete source records are needed to split.")
    try:
        train_val_ids, test_ids = train_test_split(
            source_ids, test_size=test_size, random_state=seed, shuffle=True
        )
        val_fraction = val_size / (1.0 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_fraction, random_state=seed, shuffle=True
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot create nonempty train/validation/test partitions from "
            f"{len(source_ids)} usable source records with test_size={test_size} "
            f"and val_size={val_size}; adjust the dataset size or proportions."
        ) from exc
    _validate_partition_ids(train_ids, val_ids, test_ids, source_ids)

    partitions = []
    for ids in (set(train_ids), set(val_ids), set(test_ids)):
        partition = [example for example in examples if example["source_id"] in ids]
        partitions.append(partition)
    if sum(map(len, partitions)) != len(examples):
        raise ValueError("Split partition sizes do not cover the generated examples.")
    return partitions[0], partitions[1], partitions[2]


def dataset_fingerprint(records: Sequence[dict[str, Any]]) -> str:
    """Return a stable SHA-256 fingerprint of the parsed source records."""
    canonical = json.dumps(
        records, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_split_manifest(
    path: str | Path,
    train_ids: Sequence[int],
    val_ids: Sequence[int],
    test_ids: Sequence[int],
    *,
    seed: int,
    test_size: float,
    val_size: float,
    records: Sequence[dict[str, Any]],
    usable_source_ids: Sequence[int],
    replace: bool = False,
) -> dict[str, Any]:
    """Persist source assignments; existing manifests require explicit replacement."""
    manifest_path = Path(path)
    _validate_proportions(test_size, val_size)
    expected_ids = list(usable_source_ids)
    _validate_partition_ids(train_ids, val_ids, test_ids, expected_ids)
    if manifest_path.exists() and not replace:
        raise FileExistsError(
            f"Split manifest already exists: {manifest_path}. "
            "Use explicit replacement or choose a new path."
        )
    manifest = {
        "format_version": MANIFEST_VERSION,
        "dataset_fingerprint": dataset_fingerprint(records),
        "source_record_count": len(records),
        "usable_source_ids": sorted(expected_ids),
        "seed": seed,
        "proportions": {
            "train": 1.0 - test_size - val_size,
            "validation": val_size,
            "test": test_size,
        },
        "train_source_ids": sorted(train_ids),
        "validation_source_ids": sorted(val_ids),
        "test_source_ids": sorted(test_ids),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split_manifest(path: str | Path) -> dict[str, Any]:
    """Load and structurally validate a split manifest."""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON split manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Split manifest must contain a JSON object.")
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported split manifest version {manifest.get('format_version')!r}; "
            f"expected {MANIFEST_VERSION}."
        )
    keys = ("train_source_ids", "validation_source_ids", "test_source_ids")
    if any(not isinstance(manifest.get(key), list) for key in keys):
        raise ValueError("Split manifest must contain all three source-ID lists.")
    ids = [manifest[key] for key in keys]
    for name, values in zip(("train", "validation", "test"), ids):
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError(f"The {name} source-ID list must contain integers.")
    train_ids, val_ids, test_ids = ids
    if any(len(values) != len(set(values)) for values in ids):
        raise ValueError("Split manifest source-ID lists must not contain duplicates.")
    if (set(train_ids) & set(val_ids) or set(train_ids) & set(test_ids)
            or set(val_ids) & set(test_ids)):
        raise ValueError("Split manifest source-ID partitions overlap.")
    return manifest


def create_or_load_splits(
    examples: Sequence[dict[str, Any]],
    records: Sequence[dict[str, Any]],
    manifest_path: str | Path = "data/splits/qa_split_seed42.json",
    *,
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
    mode: str = "auto",
    replace: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create or reuse splits, rejecting any manifest incompatible with the data."""
    _validate_proportions(test_size, val_size)
    if mode not in {"auto", "create", "reuse"}:
        raise ValueError("mode must be one of: auto, create, reuse.")
    source_ids = validate_examples(examples)
    path = Path(manifest_path)
    exists = path.exists()
    should_create = mode == "create" or (mode == "auto" and not exists)
    if mode == "reuse" and not exists:
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Create it explicitly first."
        )
    if should_create:
        train, val, test = split_by_source(examples, test_size, val_size, seed)
        train_ids = sorted({item["source_id"] for item in train})
        val_ids = sorted({item["source_id"] for item in val})
        test_ids = sorted({item["source_id"] for item in test})
        save_split_manifest(
            path, train_ids, val_ids, test_ids, seed=seed, test_size=test_size,
            val_size=val_size, records=records, usable_source_ids=source_ids,
            replace=replace,
        )
        return train, val, test
    if mode == "create" and exists and not replace:
        raise FileExistsError(
            f"Split manifest already exists: {path}. Use --replace-split-manifest "
            "only when intentionally replacing its assignments."
        )

    manifest = load_split_manifest(path)
    if manifest.get("source_record_count") != len(records):
        raise ValueError("Split manifest source-record count does not match the dataset.")
    if manifest.get("usable_source_ids") != source_ids:
        raise ValueError(
            "Split manifest usable source IDs do not match the preprocessed dataset."
        )
    if manifest.get("dataset_fingerprint") != dataset_fingerprint(records):
        raise ValueError(
            "Split manifest fingerprint does not match this dataset. "
            "Choose another manifest path or explicitly create and replace the manifest."
        )
    if manifest.get("seed") != seed:
        raise ValueError("Split manifest seed does not match --seed.")
    proportions = manifest.get("proportions")
    expected_proportions = {
        "train": 1.0 - test_size - val_size,
        "validation": val_size,
        "test": test_size,
    }
    if not isinstance(proportions, dict) or any(
        not isinstance(proportions.get(key), (int, float))
        or abs(proportions[key] - value) > 1e-12
        for key, value in expected_proportions.items()
    ):
        raise ValueError("Split manifest proportions do not match the requested split.")
    train_ids, val_ids, test_ids = (
        manifest["train_source_ids"],
        manifest["validation_source_ids"],
        manifest["test_source_ids"],
    )
    _validate_partition_ids(train_ids, val_ids, test_ids, source_ids)
    assignments = {
        **{source_id: "train" for source_id in train_ids},
        **{source_id: "validation" for source_id in val_ids},
        **{source_id: "test" for source_id in test_ids},
    }
    partitions = tuple(
        [example for example in examples if assignments[example["source_id"]] == name]
        for name in ("train", "validation", "test")
    )
    if sum(map(len, partitions)) != len(examples):
        raise ValueError("Manifest split sizes do not cover all generated examples.")
    return partitions


def tokenize_example(
    tokenizer: Any,
    knowledge: str,
    question: str,
    answer: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize formatted input, truncating knowledge before Q/A content."""
    encoding, _ = tokenize_example_with_stats(
        tokenizer, knowledge, question, answer, max_length
    )
    return encoding


def tokenize_example_with_stats(
    tokenizer: Any,
    knowledge: str,
    question: str,
    answer: str,
    max_length: int,
) -> tuple[dict[str, list[int]], dict[str, bool]]:
    """Return model inputs and whether knowledge or Q/A content was truncated.

    If question and answer exceed the available length, the candidate answer is
    preserved first; the question is truncated from its end. If the answer alone
    is too long, it is truncated from its end as the final fallback.
    """
    tokenizer_limit = getattr(tokenizer, "model_max_length", max_length)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        max_length = min(max_length, tokenizer_limit)
    if max_length < 1:
        raise ValueError("max_length must be greater than zero.")

    text = (
        f"Knowledge: {knowledge}\nQuestion: {question}\n"
        f"Candidate answer: {answer}"
    )
    full = tokenizer(
        text, add_special_tokens=True, truncation=False, padding=False
    )
    input_ids = list(full["input_ids"])
    if len(input_ids) <= max_length:
        return ({
            key: list(value) for key, value in full.items()
            if key in tokenizer.model_input_names
        }, {"knowledge_truncated": False, "question_or_answer_truncated": False})

    def encode(part: str) -> list[int]:
        return tokenizer.encode(part, add_special_tokens=False)

    knowledge_header = encode("Knowledge: ")
    knowledge_tokens = encode(knowledge)
    question_header = encode("\nQuestion: ")
    question_tokens = encode(question)
    answer_header = encode("\nCandidate answer: ")
    answer_tokens = encode(answer)
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    fixed = len(knowledge_header) + len(question_header) + len(answer_header)
    content_budget = max_length - special_count - fixed
    if content_budget < 0:
        raise ValueError(
            "max_length is too small for required input markers and special tokens."
        )

    original_query_size = len(question_tokens) + len(answer_tokens)
    kept_answer = answer_tokens[:content_budget]
    remaining = content_budget - len(kept_answer)
    kept_question = question_tokens[:remaining]
    remaining -= len(kept_question)
    kept_knowledge = knowledge_tokens[:remaining]
    content_ids = (
        knowledge_header + kept_knowledge + question_header + kept_question
        + answer_header + kept_answer
    )
    input_ids = tokenizer.build_inputs_with_special_tokens(content_ids)
    encoded = {"input_ids": input_ids}
    if "token_type_ids" in tokenizer.model_input_names:
        encoded["token_type_ids"] = tokenizer.create_token_type_ids_from_sequences(
            content_ids
        )
    if "attention_mask" in tokenizer.model_input_names:
        encoded["attention_mask"] = [1] * len(input_ids)
    if len(input_ids) > max_length:
        raise ValueError("Tokenizer special-token construction exceeded max_length.")
    query_truncated = (
        len(kept_question) + len(kept_answer) < original_query_size
    )
    return encoded, {
        "knowledge_truncated": len(kept_knowledge) < len(knowledge_tokens),
        "question_or_answer_truncated": query_truncated,
    }
