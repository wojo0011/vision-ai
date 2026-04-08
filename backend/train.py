from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import tensorflow as tf
from tensorflow import keras

from .dataset import summarize_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "training" / "images"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class TrainConfig:
    data_dir: Path
    models_dir: Path
    epochs: int
    batch_size: int
    image_size: int
    seed: int
    validation_split: float
    learning_rate: float
    export_name: str | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Train an image classifier from images stored in training/images."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing one subfolder per class.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory where trained models will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Mini-batch size for training."
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=180,
        help="Square image size used for resizing during training.",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--export-name",
        type=str,
        default=None,
        help="Optional folder name under models/. Defaults to a timestamped run name.",
    )
    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir.resolve(),
        models_dir=args.models_dir.resolve(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
        export_name=args.export_name,
    )


def validate_inputs(config: TrainConfig) -> None:
    if not config.data_dir.exists():
        raise FileNotFoundError(
            f"Training image directory not found: {config.data_dir}\n"
            "Create class-specific folders inside training/images before training."
        )

    dataset_summary = summarize_dataset(config.data_dir)
    if dataset_summary["label_count"] < 2:
        raise ValueError(
            "Expected at least two class folders inside training/images. "
            f"Found {dataset_summary['label_count']}."
        )

    if config.validation_split <= 0 or config.validation_split >= 1:
        raise ValueError("--validation-split must be between 0 and 1.")

    if config.epochs < 1:
        raise ValueError("--epochs must be at least 1.")

    if config.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    if config.image_size < 32:
        raise ValueError("--image-size must be at least 32.")


def build_datasets(config: TrainConfig) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    common_kwargs = {
        "directory": str(config.data_dir),
        "validation_split": config.validation_split,
        "seed": config.seed,
        "image_size": (config.image_size, config.image_size),
        "batch_size": config.batch_size,
    }

    train_ds = keras.utils.image_dataset_from_directory(
        subset="training",
        **common_kwargs,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        subset="validation",
        **common_kwargs,
    )

    class_names = list(train_ds.class_names)

    train_ds = train_ds.cache().shuffle(1000, seed=config.seed).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def build_model(image_size: int, num_classes: int) -> keras.Model:
    return keras.Sequential(
        [
            keras.Input(shape=(image_size, image_size, 3)),
            keras.layers.Rescaling(1.0 / 255),
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.2),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def create_run_dir(config: TrainConfig) -> Path:
    run_name = config.export_name or datetime.now().strftime("classifier-%Y%m%d-%H%M%S")
    run_dir = config.models_dir / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Model output directory already exists: {run_dir}\n"
            "Choose a different --export-name or remove the existing folder first."
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(config: TrainConfig, dataset_summary: dict[str, object] | None = None) -> Path:
    validate_inputs(config)
    config.models_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(config.seed)
    dataset_summary = dataset_summary or summarize_dataset(config.data_dir)

    train_ds, val_ds, class_names = build_datasets(config)
    run_dir = create_run_dir(config)

    model = build_model(config.image_size, len(class_names))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=run_dir / "best.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    training_started_at = datetime.now()
    training_started_perf = perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    training_finished_at = datetime.now()
    training_duration_seconds = perf_counter() - training_started_perf

    classifier_path = run_dir / "classifier.keras"
    labels_path = run_dir / "labels.json"
    history_path = run_dir / "history.json"
    metadata_path = run_dir / "metadata.json"

    model.save(classifier_path)
    write_json(labels_path, class_names)
    write_json(history_path, history.history)
    write_json(
        metadata_path,
        {
            "training_started_at": training_started_at.isoformat(),
            "trained_at": training_finished_at.isoformat(),
            "training_duration_seconds": round(training_duration_seconds, 3),
            "class_names": class_names,
            "dataset": dataset_summary,
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config).items()
            },
            "artifacts": {
                "classifier": str(classifier_path),
                "best_checkpoint": str(run_dir / "best.keras"),
                "labels": str(labels_path),
                "history": str(history_path),
            },
            "final_metrics": {
                key: float(values[-1]) for key, values in history.history.items() if values
            },
        },
    )

    return run_dir


def main() -> None:
    config = parse_args()
    run_dir = train(config)
    print(f"Training complete. Model artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
