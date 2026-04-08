from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

from .dataset import (
    append_training_log,
    build_dataset_status,
    summarize_dataset,
)
from .inference import list_available_models
from .train import TrainConfig, train

logger = logging.getLogger(__name__)

_STATE_LOCK = Lock()
_RUNTIME_STATE: dict[str, Any] = {
    "in_progress": False,
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "requested_fingerprint": None,
    "last_result": None,
    "last_error": None,
}


def get_runtime_state() -> dict[str, Any]:
    with _STATE_LOCK:
        return copy.deepcopy(_RUNTIME_STATE)


def _update_runtime_state(**changes: Any) -> None:
    with _STATE_LOCK:
        _RUNTIME_STATE.update(changes)


def get_dataset_status() -> dict[str, Any]:
    existing_model_names = {record.name for record in list_available_models()}
    return build_dataset_status(
        runtime_state=get_runtime_state(),
        existing_model_names=existing_model_names,
    )


def _training_log_entry(
    *,
    run_dir: Path,
    config: TrainConfig,
    dataset_summary: dict[str, Any],
    training_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "completed",
        "training_started_at": training_metadata.get("training_started_at"),
        "trained_at": training_metadata.get("trained_at") or datetime.now().isoformat(),
        "training_duration_seconds": training_metadata.get("training_duration_seconds"),
        "model_name": run_dir.name,
        "run_dir": str(run_dir),
        "dataset": dataset_summary,
        "config": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "seed": config.seed,
            "validation_split": config.validation_split,
            "learning_rate": config.learning_rate,
            "data_dir": str(config.data_dir),
            "models_dir": str(config.models_dir),
        },
    }


def _read_training_metadata(run_dir: Path) -> dict[str, Any]:
    metadata_path = run_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Could not parse metadata.json for %s", run_dir)
        return {}


def _run_training_job(config: TrainConfig, dataset_summary: dict[str, Any], job_id: str) -> None:
    try:
        run_dir = train(config, dataset_summary=dataset_summary)
        training_metadata = _read_training_metadata(run_dir)
        log_entry = _training_log_entry(
            run_dir=run_dir,
            config=config,
            dataset_summary=dataset_summary,
            training_metadata=training_metadata,
        )
        append_training_log(log_entry, config.models_dir)
        _update_runtime_state(
            in_progress=False,
            finished_at=datetime.now().isoformat(),
            last_error=None,
            last_result={
                "job_id": job_id,
                "model_name": run_dir.name,
                "run_dir": str(run_dir),
                "trained_fingerprint": dataset_summary.get("fingerprint"),
                "training_duration_seconds": training_metadata.get("training_duration_seconds"),
            },
        )
    except Exception as exc:
        logger.exception("Training job failed")
        _update_runtime_state(
            in_progress=False,
            finished_at=datetime.now().isoformat(),
            last_error=str(exc),
        )


def start_training_job(config: TrainConfig) -> dict[str, Any]:
    status = get_dataset_status()
    if status["training"]["runtime"].get("in_progress"):
        raise RuntimeError("Training is already in progress.")
    if not status["training"]["can_train"]:
        reason = status["training"]["reason"] or "The current dataset is not ready to train."
        raise RuntimeError(reason)

    dataset_summary = summarize_dataset(config.data_dir)
    job_id = str(uuid4())
    _update_runtime_state(
        in_progress=True,
        job_id=job_id,
        started_at=datetime.now().isoformat(),
        finished_at=None,
        requested_fingerprint=dataset_summary.get("fingerprint"),
        last_error=None,
    )

    worker = Thread(
        target=_run_training_job,
        kwargs={
            "config": config,
            "dataset_summary": dataset_summary,
            "job_id": job_id,
        },
        daemon=True,
    )
    worker.start()

    return {
        "job_id": job_id,
        "dataset_fingerprint": dataset_summary.get("fingerprint"),
        "message": "Training started.",
    }
