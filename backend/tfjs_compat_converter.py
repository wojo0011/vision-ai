from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from typing import Sequence


# tensorflowjs==3.x references removed NumPy aliases (np.object / np.bool).
# Recreate aliases before importing tensorflowjs modules.
def _patch_numpy_aliases() -> None:
    import numpy as np

    if "object" not in np.__dict__:
        setattr(np, "object", object)
    if "bool" not in np.__dict__:
        setattr(np, "bool", np.bool_)


def _patch_optional_imports() -> None:
    # tensorflowjs imports this unconditionally, even when conversion mode does
    # not use TF-Decision-Forests.
    if "tensorflow_decision_forests" not in sys.modules:
        sys.modules["tensorflow_decision_forests"] = types.ModuleType(
            "tensorflow_decision_forests"
        )

    # tensorflowjs also imports JAX conversion modules unconditionally.
    if "jax" not in sys.modules:
        jax_module = types.ModuleType("jax")
        jax_experimental = types.ModuleType("jax.experimental")
        jax2tf_module = types.ModuleType("jax.experimental.jax2tf")
        jax_monitoring_module = types.ModuleType("jax.monitoring")
        jax_experimental.jax2tf = jax2tf_module  # type: ignore[attr-defined]
        jax_module.experimental = jax_experimental
        jax_module.monitoring = jax_monitoring_module  # type: ignore[attr-defined]
        sys.modules["jax"] = jax_module
        sys.modules["jax.experimental"] = jax_experimental
        sys.modules["jax.experimental.jax2tf"] = jax2tf_module
        sys.modules["jax.monitoring"] = jax_monitoring_module


def _read_input_format(args: Sequence[str]) -> str | None:
    for index, value in enumerate(args):
        if value.startswith("--input_format="):
            return value.split("=", 1)[1]
        if value == "--input_format" and index + 1 < len(args):
            return args[index + 1]
    return None


def _maybe_convert_keras_to_h5(args: list[str]) -> list[str]:
    if _read_input_format(args) != "keras":
        return args

    positional_indexes = [
        index for index, value in enumerate(args) if not value.startswith("-")
    ]
    if len(positional_indexes) < 2:
        return args

    input_index = positional_indexes[-2]
    input_path = Path(args[input_index])
    if input_path.suffix.lower() != ".keras":
        return args

    import tensorflow as tf

    temp_dir = Path(tempfile.mkdtemp(prefix="tfjs-convert-"))
    temp_h5_path = temp_dir / f"{input_path.stem}.h5"
    model = tf.keras.models.load_model(str(input_path))
    model.save(str(temp_h5_path))

    rewritten_args = list(args)
    rewritten_args[input_index] = str(temp_h5_path)
    return rewritten_args


def main() -> None:
    _patch_numpy_aliases()
    _patch_optional_imports()
    from tensorflowjs.converters.converter import convert

    convert(_maybe_convert_keras_to_h5(sys.argv[1:]))


if __name__ == "__main__":
    main()
