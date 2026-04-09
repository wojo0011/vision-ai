import os
import pathlib
import shutil
import json
from threading import Lock, Thread
import logging
import subprocess
import importlib.util

logger = logging.getLogger(__name__)

_TFJS_LOCK = Lock()
_TFJS_STATE = {
    "in_progress": False,
    "job_id": None,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
    "last_result": None,
}

_MISSING_TFJS_MSG = (
    "TensorFlow.js converter is not installed. "
    "Install it with: pip install tensorflowjs"
)

_UNSUPPORTED_TFJS_LAYERS = {"RandomFlip", "RandomRotation", "RandomZoom"}


def get_tfjs_state():
    with _TFJS_LOCK:
        return dict(_TFJS_STATE)


def _update_tfjs_state(**changes):
    with _TFJS_LOCK:
        _TFJS_STATE.update(changes)


def _normalize_tfjs_model_json(output_dir: str) -> None:
    model_json_path = pathlib.Path(output_dir) / "model.json"
    if not model_json_path.exists():
        return

    payload = json.loads(model_json_path.read_text(encoding="utf-8"))

    def patch_layers(layers):
        if not isinstance(layers, list):
            return

        filtered_layers = []

        for layer in layers:
            is_removed_layer = (
                isinstance(layer, dict)
                and layer.get("class_name") in _UNSUPPORTED_TFJS_LAYERS
            )
            if is_removed_layer:
                continue
            filtered_layers.append(layer)

        layers[:] = filtered_layers

        for layer in layers:
            if not isinstance(layer, dict):
                continue
            if layer.get("class_name") != "InputLayer":
                continue

            config = layer.get("config")
            if not isinstance(config, dict):
                continue

            if "batch_shape" in config and "batchInputShape" not in config:
                config["batchInputShape"] = config.pop("batch_shape")

            if "input_shape" in config and "inputShape" not in config:
                config["inputShape"] = config.pop("input_shape")

    model_topology = payload.get("modelTopology")
    if isinstance(model_topology, dict):
        model_config = model_topology.get("model_config")
        if isinstance(model_config, dict):
            config = model_config.get("config")
            if isinstance(config, dict):
                patch_layers(config.get("layers"))

        config = model_topology.get("config")
        if isinstance(config, dict):
            patch_layers(config.get("layers"))

    model_json_path.write_text(json.dumps(payload), encoding="utf-8")


def _run_tfjs_conversion(keras_path, output_dir, job_id):
    try:
        _update_tfjs_state(
            in_progress=True,
            job_id=job_id,
            started_at=os.times(),
            finished_at=None,
            last_error=None,
            last_result=None,
        )
        if importlib.util.find_spec("tensorflowjs") is None:
            _update_tfjs_state(
                in_progress=False,
                finished_at=os.times(),
                last_error=_MISSING_TFJS_MSG,
                last_result={
                    "job_id": job_id,
                    "keras_path": keras_path,
                    "output_dir": output_dir,
                    "stdout": "",
                    "stderr": _MISSING_TFJS_MSG,
                    "returncode": 1,
                },
            )
            return

        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        import sys
        result = subprocess.run([
            sys.executable,
            '-m',
            'backend.tfjs_compat_converter',
            '--input_format=keras', keras_path, output_dir
        ], capture_output=True, text=True)

        if result.returncode == 0:
            model_dir = pathlib.Path(keras_path).resolve().parent
            target_dir = pathlib.Path(output_dir).resolve()
            for filename in ("labels.json", "metadata.json"):
                source_file = model_dir / filename
                if source_file.exists():
                    shutil.copy2(source_file, target_dir / filename)
            _normalize_tfjs_model_json(str(target_dir))

        _update_tfjs_state(
            in_progress=False,
            finished_at=os.times(),
            last_error=None if result.returncode == 0 else result.stderr,
            last_result={
                "job_id": job_id,
                "keras_path": keras_path,
                "output_dir": output_dir,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            },
        )
    except Exception as exc:
        logger.exception("TFJS conversion failed")
        _update_tfjs_state(
            in_progress=False,
            finished_at=os.times(),
            last_error=str(exc),
        )


def start_tfjs_conversion(keras_path, output_dir, job_id):
    worker = Thread(
        target=_run_tfjs_conversion,
        args=(keras_path, output_dir, job_id),
        daemon=True,
    )
    worker.start()
    return {"job_id": job_id, "message": "TFJS conversion started."}
