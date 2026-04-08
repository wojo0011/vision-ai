from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "training" / "images"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
TRAINING_LOG_FILENAME = "training-log.json"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
INVALID_LABEL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


class DatasetError(RuntimeError):
    """Raised when dataset capture or inspection fails."""


def resolve_data_dir(data_dir: Path | None = None) -> Path:
    return (data_dir or DEFAULT_DATA_DIR).resolve()


def resolve_models_dir(models_dir: Path | None = None) -> Path:
    return (models_dir or DEFAULT_MODELS_DIR).resolve()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def training_log_path(models_dir: Path | None = None) -> Path:
    return resolve_models_dir(models_dir) / TRAINING_LOG_FILENAME


def normalize_label(raw_label: str) -> str:
    normalized = re.sub(r"\s+", " ", raw_label.strip())
    normalized = INVALID_LABEL_CHARS.sub("-", normalized)
    normalized = normalized.strip(". ").lower()
    if not normalized:
        raise DatasetError("Enter a label before saving a training image.")
    return normalized


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def metadata_sidecar_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.name}.meta.json")


def load_image_metadata(image_path: Path) -> dict[str, Any]:
    sidecar_path = metadata_sidecar_path(image_path)
    if not sidecar_path.exists():
        return {}

    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    return payload


def write_image_metadata(image_path: Path, payload: dict[str, Any]) -> None:
    metadata_sidecar_path(image_path).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def serialize_image_detail(
    image_path: Path,
    relative_path: str,
    label_name: str,
) -> dict[str, Any]:
    stat = image_path.stat()
    metadata = load_image_metadata(image_path)
    width = metadata.get("width")
    height = metadata.get("height")
    resolution = metadata.get("resolution")
    if not resolution and width and height:
        resolution = f"{width}x{height}"

    return {
        "label": metadata.get("label") or label_name,
        "relative_path": relative_path,
        "captured_at": metadata.get("captured_at")
        or datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "width": width,
        "height": height,
        "resolution": resolution,
        "file_size_bytes": metadata.get("file_size_bytes", stat.st_size),
    }


def list_label_directories(data_dir: Path | None = None) -> list[Path]:
    base_dir = resolve_data_dir(data_dir)
    if not base_dir.exists():
        return []
    return sorted(path for path in base_dir.iterdir() if path.is_dir())


def summarize_dataset(data_dir: Path | None = None) -> dict[str, Any]:
    base_dir = resolve_data_dir(data_dir)
    labels: list[dict[str, Any]] = []
    fingerprint_parts: list[str] = []
    total_images = 0

    for label_dir in list_label_directories(base_dir):
        image_paths = sorted(path for path in label_dir.rglob("*") if is_image_file(path))
        if not image_paths:
            continue
        relative_images = [str(path.relative_to(base_dir)).replace("\\", "/") for path in image_paths]
        image_details = [
            serialize_image_detail(path, relative_path, label_dir.name)
            for path, relative_path in zip(image_paths, relative_images)
        ]
        for path, relative_path in zip(image_paths, relative_images):
            stat = path.stat()
            fingerprint_parts.append(f"{relative_path}|{stat.st_size}|{stat.st_mtime_ns}")

        labels.append(
            {
                "name": label_dir.name,
                "directory": str(label_dir),
                "image_count": len(image_paths),
                "images": relative_images,
                "image_details": image_details,
            }
        )
        total_images += len(image_paths)

    fingerprint = None
    if fingerprint_parts:
        digest = hashlib.sha256()
        for part in fingerprint_parts:
            digest.update(part.encode("utf-8"))
        fingerprint = digest.hexdigest()

    return {
        "data_dir": str(base_dir),
        "label_count": len(labels),
        "total_images": total_images,
        "labels": labels,
        "fingerprint": fingerprint,
        "has_images": total_images > 0,
        "has_multiple_labels": len(labels) >= 2,
    }


def read_training_log(models_dir: Path | None = None) -> list[dict[str, Any]]:
    log_path = training_log_path(models_dir)
    if not log_path.exists():
        return []

    try:
        payload = json.loads(log_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid training log JSON in {log_path}") from exc

    if not isinstance(payload, list):
        raise DatasetError(f"Expected a list in training log {log_path}")

    return payload


def write_training_log(entries: list[dict[str, Any]], models_dir: Path | None = None) -> None:
    log_path = training_log_path(models_dir)
    ensure_directory(log_path.parent)
    log_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def append_training_log(entry: dict[str, Any], models_dir: Path | None = None) -> None:
    entries = read_training_log(models_dir)
    entries.insert(0, entry)
    write_training_log(entries, models_dir)


def latest_completed_log(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((entry for entry in entries if entry.get("status") == "completed"), None)


def summarize_training_state(
    dataset_summary: dict[str, Any],
    training_logs: list[dict[str, Any]],
    runtime_state: dict[str, Any] | None = None,
    existing_model_names: set[str] | None = None,
) -> dict[str, Any]:
    runtime_state = runtime_state or {}
    latest_log = latest_completed_log(training_logs)
    available_logs = training_logs
    if existing_model_names is not None:
        available_logs = [
            entry
            for entry in training_logs
            if entry.get("status") == "completed" and entry.get("model_name") in existing_model_names
        ]
    latest_available_log = latest_completed_log(available_logs)
    current_fingerprint = dataset_summary.get("fingerprint")
    latest_fingerprint = (
        latest_available_log.get("dataset", {}).get("fingerprint")
        if latest_available_log
        else None
    )
    already_trained = bool(
        current_fingerprint and latest_fingerprint and current_fingerprint == latest_fingerprint
    )

    can_train = (
        dataset_summary.get("has_images", False)
        and dataset_summary.get("has_multiple_labels", False)
        and not already_trained
        and not runtime_state.get("in_progress", False)
    )

    train_reason = None
    if runtime_state.get("in_progress"):
        train_reason = "Training is already in progress."
    elif not dataset_summary.get("has_images", False):
        train_reason = "Add at least one image before training."
    elif not dataset_summary.get("has_multiple_labels", False):
        train_reason = "Training needs images for at least two labels."
    elif already_trained:
        train_reason = "The current image set already matches the latest trained model."

    return {
        "can_train": can_train,
        "is_current_dataset_trained": already_trained,
        "latest_completed_log": latest_log,
        "latest_available_model_log": latest_available_log,
        "latest_trained_fingerprint": latest_fingerprint,
        "reason": train_reason,
        "runtime": runtime_state,
        "log_count": len(training_logs),
    }


def build_dataset_status(
    data_dir: Path | None = None,
    models_dir: Path | None = None,
    runtime_state: dict[str, Any] | None = None,
    existing_model_names: set[str] | None = None,
) -> dict[str, Any]:
    dataset_summary = summarize_dataset(data_dir)
    training_logs = read_training_log(models_dir)
    training_summary = summarize_training_state(
        dataset_summary,
        training_logs,
        runtime_state,
        existing_model_names=existing_model_names,
    )
    return {
        "dataset": dataset_summary,
        "training": training_summary,
        "logs": training_logs,
    }


def save_labeled_image(
    image_bytes: bytes,
    label: str,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    if not image_bytes:
        raise DatasetError("The request body did not contain any image data.")

    normalized_label = normalize_label(label)
    base_dir = ensure_directory(resolve_data_dir(data_dir))
    label_dir = ensure_directory(base_dir / normalized_label)

    try:
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    except tf.errors.InvalidArgumentError as exc:
        raise DatasetError("The uploaded frame could not be decoded as an image.") from exc

    image.set_shape([None, None, 3])
    encoded_image = tf.io.encode_jpeg(image, quality=95, optimize_size=True)
    filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.jpg"
    image_path = label_dir / filename
    tf.io.write_file(str(image_path), encoded_image)
    file_size_bytes = image_path.stat().st_size
    height = int(image.shape[0])
    width = int(image.shape[1])
    captured_at = datetime.now().isoformat()
    metadata = {
        "label": normalized_label,
        "captured_at": captured_at,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "file_size_bytes": file_size_bytes,
    }
    write_image_metadata(image_path, metadata)

    return {
        "label": normalized_label,
        "relative_path": str(image_path.relative_to(base_dir)).replace("\\", "/"),
        "path": str(image_path),
        **metadata,
    }


def resolve_relative_image_path(relative_path: str, data_dir: Path | None = None) -> Path:
    base_dir = resolve_data_dir(data_dir)
    if not relative_path.strip():
        raise DatasetError("An image path is required.")

    normalized = Path(relative_path.replace("\\", "/"))
    if normalized.is_absolute() or ".." in normalized.parts:
        raise DatasetError("Invalid image path.")

    image_path = (base_dir / normalized).resolve()
    if image_path == base_dir or base_dir not in image_path.parents:
        raise DatasetError("Invalid image path.")
    if not image_path.exists() or not is_image_file(image_path):
        raise DatasetError(f"Training image '{relative_path}' was not found.")

    return image_path


def guess_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(path.name)
    return media_type or "application/octet-stream"


def _remove_empty_directories(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    while current != stop_dir and current.exists():
        try:
            next(current.iterdir())
            break
        except StopIteration:
            current.rmdir()
            current = current.parent


def _unique_target_path(target_path: Path) -> Path:
    if not target_path.exists():
        return target_path

    counter = 1
    while True:
        candidate = target_path.with_name(f"{target_path.stem}-{counter}{target_path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def delete_training_image(relative_path: str, data_dir: Path | None = None) -> dict[str, Any]:
    base_dir = resolve_data_dir(data_dir)
    image_path = resolve_relative_image_path(relative_path, base_dir)
    old_label = image_path.parent.name
    deleted_path = str(image_path.relative_to(base_dir)).replace("\\", "/")
    sidecar_path = metadata_sidecar_path(image_path)
    image_path.unlink()
    if sidecar_path.exists():
        sidecar_path.unlink()
    _remove_empty_directories(image_path.parent, base_dir)

    return {
        "label": old_label,
        "relative_path": deleted_path,
    }


def relabel_training_image(
    relative_path: str,
    new_label: str,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    base_dir = resolve_data_dir(data_dir)
    image_path = resolve_relative_image_path(relative_path, base_dir)
    normalized_label = normalize_label(new_label)
    target_dir = ensure_directory(base_dir / normalized_label)
    target_path = target_dir / image_path.name

    if image_path.parent == target_dir:
        return {
            "old_label": image_path.parent.name,
            "new_label": normalized_label,
            "old_relative_path": str(image_path.relative_to(base_dir)).replace("\\", "/"),
            "new_relative_path": str(image_path.relative_to(base_dir)).replace("\\", "/"),
        }

    target_path = _unique_target_path(target_path)
    old_relative_path = str(image_path.relative_to(base_dir)).replace("\\", "/")
    old_parent = image_path.parent
    old_sidecar_path = metadata_sidecar_path(image_path)
    metadata = load_image_metadata(image_path)
    image_path.rename(target_path)
    new_sidecar_path = metadata_sidecar_path(target_path)
    if old_sidecar_path.exists():
        metadata["label"] = normalized_label
        old_sidecar_path.rename(new_sidecar_path)
        write_image_metadata(target_path, metadata)
    _remove_empty_directories(old_parent, base_dir)

    return {
        "old_label": old_parent.name,
        "new_label": normalized_label,
        "old_relative_path": old_relative_path,
        "new_relative_path": str(target_path.relative_to(base_dir)).replace("\\", "/"),
    }
