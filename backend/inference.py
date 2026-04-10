from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
CLASSIFIER_FILENAME = "classifier.keras"
LABELS_FILENAME = "labels.json"
METADATA_FILENAME = "metadata.json"
HISTORY_FILENAME = "history.json"
EVALUATION_FILENAME = "evaluation.json"
EVALUATION_VERSION = 3


class InferenceError(RuntimeError):
    """Raised when model discovery or inference cannot continue."""


class ModelNotFoundError(InferenceError):
    """Raised when a requested model does not exist."""


@dataclass(frozen=True)
class ModelRecord:
    name: str
    run_dir: Path
    classifier_path: Path
    labels_path: Path | None
    metadata_path: Path | None
    trained_at: str | None
    modified_at: float


@dataclass(frozen=True)
class LoadedModel:
    record: ModelRecord
    model: keras.Model
    labels: list[str]
    image_size: int
    metadata: dict[str, Any]


def resolve_models_dir(models_dir: Path | None = None) -> Path:
    if models_dir is not None:
        return models_dir.resolve()
    override = os.getenv("VISION_AI_MODELS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_MODELS_DIR


def _safe_read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Invalid JSON in {path}") from exc


def _safe_write_json(path: Path, payload: object) -> None:
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        # Serialization should still succeed even if we cannot persist the cache file.
        pass


def _series_metric(values: Any) -> list[float]:
    if not isinstance(values, list) or not values:
        return []

    series: list[float] = []
    for value in values:
        try:
            series.append(float(value))
        except (TypeError, ValueError):
            continue
    return series


def _has_confusion_matrix(evaluation: Any) -> bool:
    if not isinstance(evaluation, dict):
        return False

    confusion_matrix = evaluation.get("confusion_matrix")
    if not isinstance(confusion_matrix, dict):
        return False

    labels = confusion_matrix.get("labels")
    counts = confusion_matrix.get("counts")
    if not isinstance(labels, list) or not isinstance(counts, list) or not labels:
        return False

    if len(labels) != len(counts):
        return False

    for row in counts:
        if not isinstance(row, list) or len(row) != len(labels):
            return False
        for value in row:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                return False
            if not numeric_value.is_integer() or numeric_value < 0:
                return False

    return True


def _summarize_per_class_accuracy(
    model: keras.Model,
    val_ds: tf.data.Dataset,
    class_names: list[str],
) -> dict[str, Any]:
    total_counts = [0] * len(class_names)
    correct_counts = [0] * len(class_names)
    confusion_counts = [[0] * len(class_names) for _ in class_names]

    for images, labels in val_ds:
        predictions = model(images, training=False)
        predicted_labels = tf.argmax(predictions, axis=1, output_type=tf.int32).numpy().tolist()
        true_labels = tf.cast(labels, tf.int32).numpy().tolist()

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            class_index = int(true_label)
            predicted_index = int(predicted_label)
            if class_index < 0 or class_index >= len(class_names):
                continue
            total_counts[class_index] += 1
            if 0 <= predicted_index < len(class_names):
                confusion_counts[class_index][predicted_index] += 1
            if predicted_index == class_index:
                correct_counts[class_index] += 1

    per_class_accuracy = []
    for index, class_name in enumerate(class_names):
        sample_count = total_counts[index]
        accuracy = correct_counts[index] / sample_count if sample_count else None
        per_class_accuracy.append(
            {
                "class_name": class_name,
                "accuracy": float(accuracy) if accuracy is not None else None,
                "sample_count": sample_count,
                "correct_count": correct_counts[index],
            }
        )

    total_samples = sum(total_counts)
    total_correct = sum(correct_counts)
    validation_accuracy = total_correct / total_samples if total_samples else None
    predicted_totals = [
        sum(confusion_counts[row_index][column_index] for row_index in range(len(class_names)))
        for column_index in range(len(class_names))
    ]

    return {
        "version": EVALUATION_VERSION,
        "split_source": "keras.image_dataset_from_directory",
        "shuffle": True,
        "validation_sample_count": total_samples,
        "validation_correct_count": total_correct,
        "validation_accuracy": float(validation_accuracy) if validation_accuracy is not None else None,
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": {
            "labels": [str(class_name) for class_name in class_names],
            "counts": confusion_counts,
            "row_totals": total_counts,
            "column_totals": predicted_totals,
        },
    }


def _resolve_evaluation(record: ModelRecord, metadata: dict[str, Any]) -> dict[str, Any]:
    evaluation_path = record.run_dir / EVALUATION_FILENAME
    evaluation = _safe_read_json(evaluation_path if evaluation_path.exists() else None)
    metadata_evaluation = metadata.get("evaluation")
    legacy_fallback: dict[str, Any] = {}
    if isinstance(evaluation, dict) and evaluation.get("per_class_accuracy"):
        legacy_fallback = evaluation
    elif isinstance(metadata_evaluation, dict) and metadata_evaluation.get("per_class_accuracy"):
        legacy_fallback = metadata_evaluation

    if (
        isinstance(evaluation, dict)
        and evaluation.get("version") == EVALUATION_VERSION
        and evaluation.get("per_class_accuracy")
        and _has_confusion_matrix(evaluation)
    ):
        return evaluation

    if (
        isinstance(metadata_evaluation, dict)
        and metadata_evaluation.get("version") == EVALUATION_VERSION
        and metadata_evaluation.get("per_class_accuracy")
        and _has_confusion_matrix(metadata_evaluation)
    ):
        if (
            not evaluation_path.exists()
            or evaluation.get("version") != EVALUATION_VERSION
            or not _has_confusion_matrix(evaluation)
        ):
            _safe_write_json(evaluation_path, metadata_evaluation)
        return metadata_evaluation

    config = metadata.get("config", {})
    class_names = metadata.get("class_names")
    if not isinstance(config, dict) or not isinstance(class_names, list) or not class_names:
        return legacy_fallback

    data_dir_value = config.get("data_dir")
    if not data_dir_value:
        return legacy_fallback

    data_dir = Path(str(data_dir_value)).expanduser()
    if not data_dir.exists():
        return legacy_fallback

    try:
        validation_split = float(config.get("validation_split", 0.2))
        seed = int(config.get("seed", 123))
        image_size = int(config.get("image_size", 180))
        batch_size = int(config.get("batch_size", 32))
    except (TypeError, ValueError):
        return legacy_fallback

    dataset_info = metadata.get("dataset")
    if not isinstance(dataset_info, dict):
        return legacy_fallback

    try:
        sample_paths: list[Path] = []
        sample_labels: list[int] = []
        class_index_by_name = {
            str(class_name): index for index, class_name in enumerate(class_names)
        }

        for label_info in dataset_info.get("labels", []):
            if not isinstance(label_info, dict):
                continue
            class_name = str(label_info.get("name", ""))
            if class_name not in class_index_by_name:
                continue
            label_index = class_index_by_name[class_name]
            for relative_path in label_info.get("images", []):
                sample_paths.append(data_dir / str(relative_path))
                sample_labels.append(label_index)

        if not sample_paths:
            return legacy_fallback

        rng = np.random.RandomState(seed)
        sample_paths = list(sample_paths)
        sample_labels = list(sample_labels)
        rng.shuffle(sample_paths)
        rng = np.random.RandomState(seed)
        rng.shuffle(sample_labels)

        num_val_samples = int(validation_split * len(sample_paths))
        if num_val_samples <= 0:
            return legacy_fallback

        validation_paths = sample_paths[-num_val_samples:]
        validation_labels = sample_labels[-num_val_samples:]
        model = _load_model_from_path(str(record.classifier_path))

        total_counts = [0] * len(class_names)
        correct_counts = [0] * len(class_names)
        confusion_counts = [[0] * len(class_names) for _ in class_names]

        for batch_start in range(0, len(validation_paths), batch_size):
            batch_paths = validation_paths[batch_start: batch_start + batch_size]
            batch_labels = validation_labels[batch_start: batch_start + batch_size]
            batch_images = []

            for image_path in batch_paths:
                image_bytes = tf.io.read_file(str(image_path))
                image = tf.io.decode_image(
                    image_bytes,
                    channels=3,
                    expand_animations=False,
                )
                image.set_shape([None, None, 3])
                batch_images.append(
                    tf.image.resize(image, (image_size, image_size))
                )

            if not batch_images:
                continue

            batch_tensor = tf.cast(tf.stack(batch_images, axis=0), tf.float32)
            predictions = model(batch_tensor, training=False)
            predicted_labels = tf.argmax(
                predictions, axis=1, output_type=tf.int32
            ).numpy().tolist()

            for true_label, predicted_label in zip(batch_labels, predicted_labels):
                class_index = int(true_label)
                predicted_index = int(predicted_label)
                if class_index < 0 or class_index >= len(class_names):
                    continue
                total_counts[class_index] += 1
                if 0 <= predicted_index < len(class_names):
                    confusion_counts[class_index][predicted_index] += 1
                if predicted_index == class_index:
                    correct_counts[class_index] += 1

        per_class_accuracy = []
        for index, class_name in enumerate(class_names):
            sample_count = total_counts[index]
            accuracy = correct_counts[index] / sample_count if sample_count else None
            per_class_accuracy.append(
                {
                    "class_name": str(class_name),
                    "accuracy": float(accuracy) if accuracy is not None else None,
                    "sample_count": sample_count,
                    "correct_count": correct_counts[index],
                }
            )

        total_samples = sum(total_counts)
        total_correct = sum(correct_counts)
        validation_accuracy = total_correct / total_samples if total_samples else None
        predicted_totals = [
            sum(confusion_counts[row_index][column_index] for row_index in range(len(class_names)))
            for column_index in range(len(class_names))
        ]
        evaluation = {
            "version": EVALUATION_VERSION,
            "split_source": "keras.image_dataset_from_directory",
            "shuffle": True,
            "validation_sample_count": total_samples,
            "validation_correct_count": total_correct,
            "validation_accuracy": float(validation_accuracy) if validation_accuracy is not None else None,
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": {
                "labels": [str(class_name) for class_name in class_names],
                "counts": confusion_counts,
                "row_totals": total_counts,
                "column_totals": predicted_totals,
            },
        }
        _safe_write_json(evaluation_path, evaluation)
        return evaluation
    except Exception:
        return legacy_fallback


def list_available_models(models_dir: Path | None = None) -> list[ModelRecord]:
    base_dir = (models_dir or resolve_models_dir()).resolve()
    if not base_dir.exists():
        return []

    records: list[ModelRecord] = []
    for run_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        classifier_path = run_dir / CLASSIFIER_FILENAME
        if not classifier_path.exists():
            continue

        labels_path = run_dir / LABELS_FILENAME
        metadata_path = run_dir / METADATA_FILENAME
        metadata = _safe_read_json(metadata_path)
        trained_at = metadata.get("trained_at")
        modified_at = classifier_path.stat().st_mtime
        records.append(
            ModelRecord(
                name=run_dir.name,
                run_dir=run_dir,
                classifier_path=classifier_path,
                labels_path=labels_path if labels_path.exists() else None,
                metadata_path=metadata_path if metadata_path.exists() else None,
                trained_at=trained_at,
                modified_at=modified_at,
            )
        )

    def sort_key(record: ModelRecord) -> tuple[datetime, float]:
        if record.trained_at:
            try:
                parsed = datetime.fromisoformat(record.trained_at)
                return parsed, record.modified_at
            except ValueError:
                pass
        return datetime.fromtimestamp(record.modified_at), record.modified_at

    return sorted(records, key=sort_key, reverse=True)


def serialize_model_record(record: ModelRecord) -> dict[str, Any]:
    metadata = _safe_read_json(record.metadata_path)
    history_path = record.run_dir / HISTORY_FILENAME
    history = _safe_read_json(history_path if history_path.exists() else None)
    evaluation = _resolve_evaluation(record, metadata)

    def _best_metric(values: Any) -> float | None:
        if not isinstance(values, list) or not values:
            return None
        try:
            return float(max(values))
        except (TypeError, ValueError):
            return None

    def _latest_metric(values: Any) -> float | None:
        if not isinstance(values, list) or not values:
            return None
        try:
            return float(values[-1])
        except (TypeError, ValueError):
            return None

    run_size_bytes = 0
    for path in record.run_dir.rglob("*"):
        if path.is_file():
            run_size_bytes += path.stat().st_size

    dataset_info = metadata.get("dataset", {})
    dataset_labels = dataset_info.get("labels", [])

    accuracy_summary = {
        "train_accuracy_final": _latest_metric(history.get("accuracy")),
        "train_accuracy_best": _best_metric(history.get("accuracy")),
        "val_accuracy_final": _latest_metric(history.get("val_accuracy")),
        "val_accuracy_best": _best_metric(history.get("val_accuracy")),
        "train_loss_final": _latest_metric(history.get("loss")),
        "val_loss_final": _latest_metric(history.get("val_loss")),
    }
    history_summary = {
        "epochs": max(
            len(_series_metric(history.get("accuracy"))),
            len(_series_metric(history.get("val_accuracy"))),
            len(_series_metric(history.get("loss"))),
            len(_series_metric(history.get("val_loss"))),
        ),
        "accuracy": _series_metric(history.get("accuracy")),
        "val_accuracy": _series_metric(history.get("val_accuracy")),
        "loss": _series_metric(history.get("loss")),
        "val_loss": _series_metric(history.get("val_loss")),
    }

    model_type = metadata.get("model_type")
    if not model_type:
        architecture = metadata.get("model_architecture") or "Sequential"
        model_type = f"Keras {architecture} CNN"

    return {
        "name": record.name,
        "run_dir": str(record.run_dir),
        "trained_at": record.trained_at,
        "training_started_at": metadata.get("training_started_at"),
        "training_duration_seconds": metadata.get("training_duration_seconds"),
        "image_size": metadata.get("config", {}).get("image_size"),
        "class_count": len(metadata.get("class_names", [])),
        "labels": metadata.get("class_names", []),
        "dataset_total_images": dataset_info.get("total_images"),
        "dataset_label_count": dataset_info.get("label_count"),
        "dataset_labels": dataset_labels,
        "run_size_bytes": run_size_bytes,
        "model_type": model_type,
        "accuracy": accuracy_summary,
        "history": history_summary,
        "evaluation": evaluation,
        "classifier_path": str(record.classifier_path),
    }


def get_model_record(model_name: str, models_dir: Path | None = None) -> ModelRecord:
    record = next(
        (item for item in list_available_models(models_dir) if item.name == model_name),
        None,
    )
    if record is None:
        raise ModelNotFoundError(f"Model '{model_name}' was not found.")
    return record


@lru_cache(maxsize=8)
def _load_model_from_path(classifier_path: str) -> keras.Model:
    return keras.models.load_model(classifier_path, compile=False)


def _load_labels(record: ModelRecord, metadata: dict[str, Any]) -> list[str]:
    if record.labels_path and record.labels_path.exists():
        labels = json.loads(record.labels_path.read_text(encoding="utf-8"))
        if isinstance(labels, list) and labels:
            return [str(label) for label in labels]

    class_names = metadata.get("class_names")
    if isinstance(class_names, list) and class_names:
        return [str(label) for label in class_names]

    raise InferenceError(
        f"Could not determine class labels for model '{record.name}'. "
        "Expected labels.json or metadata.json with class_names."
    )


def _infer_image_size(model: keras.Model, metadata: dict[str, Any]) -> int:
    configured_size = metadata.get("config", {}).get("image_size")
    if configured_size:
        return int(configured_size)

    input_shape = getattr(model, "input_shape", None)
    if isinstance(input_shape, tuple) and len(input_shape) >= 3 and input_shape[1]:
        return int(input_shape[1])

    raise InferenceError("Could not determine the expected image size for inference.")


def get_loaded_model(
    model_name: str | None = None, models_dir: Path | None = None
) -> LoadedModel:
    records = list_available_models(models_dir)
    if not records:
        raise InferenceError(
            "No trained models were found. Train a model first so the frontend can classify images."
        )

    record = records[0] if model_name is None else get_model_record(model_name, models_dir)

    metadata = _safe_read_json(record.metadata_path)
    model = _load_model_from_path(str(record.classifier_path))
    labels = _load_labels(record, metadata)
    image_size = _infer_image_size(model, metadata)

    return LoadedModel(
        record=record,
        model=model,
        labels=labels,
        image_size=image_size,
        metadata=metadata,
    )


def classify_image_bytes(
    image_bytes: bytes,
    model_name: str | None = None,
    top_k: int = 3,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    if not image_bytes:
        raise InferenceError("The request body did not contain any image data.")

    loaded = get_loaded_model(model_name=model_name, models_dir=models_dir)

    try:
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError as exc:
        raise InferenceError("The uploaded frame could not be decoded as an image.") from exc

    image.set_shape([None, None, 3])
    resized = tf.image.resize(image, (loaded.image_size, loaded.image_size))
    batch = tf.expand_dims(tf.cast(resized, tf.float32), axis=0)

    predictions = loaded.model(batch, training=False)
    scores = tf.squeeze(predictions).numpy().tolist()
    if not isinstance(scores, list):
        scores = [float(scores)]

    total = sum(scores)
    if total <= 0 or any(score < 0 for score in scores):
        probabilities = tf.nn.softmax(predictions[0]).numpy().tolist()
    else:
        probabilities = [float(score) / total for score in scores]

    ranked = sorted(
        [
            {
                "label": label,
                "confidence": float(confidence),
            }
            for label, confidence in zip(loaded.labels, probabilities)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return {
        "model": serialize_model_record(loaded.record),
        "top_prediction": ranked[0],
        "predictions": ranked[:top_k],
        "image_size": loaded.image_size,
    }


def delete_model(model_name: str, models_dir: Path | None = None) -> dict[str, Any]:
    record = get_model_record(model_name, models_dir)
    base_dir = resolve_models_dir(models_dir)
    run_dir = record.run_dir.resolve()

    if run_dir == base_dir or base_dir not in run_dir.parents:
        raise InferenceError("Refusing to delete a model outside the models directory.")

    shutil.rmtree(run_dir)
    _load_model_from_path.cache_clear()

    return {
        "deleted_model": record.name,
        "run_dir": str(run_dir),
    }
