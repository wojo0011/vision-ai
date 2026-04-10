from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .jira_rag.service import jira_rag_service
from .jira_rag.schemas import JiraChatRequest, JiraChatResponse, JiraProjectOption, JiraSyncRequest, JiraSyncResponse
from .dataset import (
    DatasetError,
    delete_training_image,
    guess_media_type,
    relabel_training_image,
    resolve_relative_image_path,
    save_labeled_image,
)
from .inference import (
    InferenceError,
    ModelNotFoundError,
    classify_image_bytes,
    delete_model,
    list_available_models,
    resolve_models_dir,
    serialize_model_record,
)
from .train import DEFAULT_DATA_DIR, DEFAULT_MODELS_DIR, TrainConfig
from .training_service import get_dataset_status, start_training_job
from .tfjs_service import get_tfjs_state, start_tfjs_conversion
import uuid

load_dotenv()

logger = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )


def _cors_origins() -> list[str]:
    raw_value = os.getenv("VISION_AI_CORS_ORIGINS")
    if not raw_value:
        return ["http://localhost:5173", "http://127.0.0.1:5173"]
    return [origin.strip() for origin in raw_value.split(",") if origin.strip()]


app = FastAPI(title="Vision AI + Jira RAG API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    epochs: int = Field(default=15, ge=1)
    batch_size: int = Field(default=32, ge=1)
    image_size: int = Field(default=180, ge=32)
    seed: int = 123
    validation_split: float = Field(default=0.2, gt=0, lt=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    export_name: str | None = None


class ImagePathRequest(BaseModel):
    relative_path: str = Field(min_length=1)


class ImageRelabelRequest(ImagePathRequest):
    label: str = Field(min_length=1)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_dir": str(resolve_models_dir()),
        "jira": jira_rag_service.health(),
    }


@app.get("/health")
def root_health() -> dict[str, Any]:
    return {
        "status": "ok",
        "vision_ai": {
            "models_dir": str(resolve_models_dir()),
        },
        "jira": jira_rag_service.health(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    try:
        available_models = list_available_models()
        return {
            "models_dir": str(resolve_models_dir()),
            "models": [serialize_model_record(record) for record in available_models],
            "has_models": bool(available_models),
        }
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while listing models")
        raise HTTPException(status_code=500, detail="Unexpected error while listing models.") from exc


@app.get("/api/dataset")
def dataset() -> dict[str, Any]:
    try:
        return get_dataset_status()
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while fetching dataset status")
        raise HTTPException(status_code=500, detail="Unexpected error while fetching dataset status.") from exc


@app.get("/api/training-image")
def training_image(path: str = Query(..., min_length=1)) -> FileResponse:
    try:
        image_path = resolve_relative_image_path(path)
        return FileResponse(image_path, media_type=guess_media_type(image_path))
    except DatasetError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while loading training image")
        raise HTTPException(status_code=500, detail="Unexpected error while loading training image.") from exc


@app.post("/api/capture")
async def capture(request: Request, label: str = Query(..., min_length=1)) -> dict[str, Any]:
    image_bytes = await request.body()
    try:
        capture_result = save_labeled_image(image_bytes=image_bytes, label=label)
        return {
            "saved": capture_result,
            "status": get_dataset_status(),
        }
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while capturing training image")
        raise HTTPException(status_code=500, detail="Unexpected error while capturing training image.") from exc


@app.patch("/api/dataset/images/label")
def relabel_image(payload: ImageRelabelRequest) -> dict[str, Any]:
    try:
        updated = relabel_training_image(payload.relative_path, payload.label)
        return {
            "updated": updated,
            "status": get_dataset_status(),
        }
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while relabeling training image")
        raise HTTPException(status_code=500, detail="Unexpected error while relabeling training image.") from exc


@app.delete("/api/dataset/images")
def delete_image(payload: ImagePathRequest) -> dict[str, Any]:
    try:
        deleted = delete_training_image(payload.relative_path)
        return {
            "deleted": deleted,
            "status": get_dataset_status(),
        }
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while deleting training image")
        raise HTTPException(status_code=500, detail="Unexpected error while deleting training image.") from exc


@app.post("/api/train")
def trigger_training(payload: TrainRequest) -> dict[str, Any]:
    try:
        config = TrainConfig(
            data_dir=DEFAULT_DATA_DIR.resolve(),
            models_dir=DEFAULT_MODELS_DIR.resolve(),
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            image_size=payload.image_size,
            seed=payload.seed,
            validation_split=payload.validation_split,
            learning_rate=payload.learning_rate,
            export_name=payload.export_name,
        )
        return start_training_job(config)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while starting training")
        raise HTTPException(status_code=500, detail="Unexpected error while starting training.") from exc


@app.delete("/api/models/{model_name}")
def remove_model(model_name: str) -> dict[str, Any]:
    try:
        deleted = delete_model(model_name)
        available_models = list_available_models()
        return {
            "deleted": deleted,
            "models": [serialize_model_record(record) for record in available_models],
            "status": get_dataset_status(),
        }
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while deleting model")
        raise HTTPException(status_code=500, detail="Unexpected error while deleting model.") from exc


@app.post("/api/classify")
async def classify(
    request: Request,
    model: str | None = None,
    top_k: int = Query(default=3, ge=1, le=10),
) -> dict[str, Any]:
    image_bytes = await request.body()
    try:
        return classify_image_bytes(image_bytes=image_bytes, model_name=model, top_k=top_k)
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while classifying image")
        raise HTTPException(status_code=500, detail="Unexpected error while classifying image.") from exc


@app.post("/api/train/tfjs")
def trigger_tfjs_training(payload: TrainRequest) -> dict[str, Any]:
    try:
        config = TrainConfig(
            data_dir=DEFAULT_DATA_DIR.resolve(),
            models_dir=DEFAULT_MODELS_DIR.resolve(),
            epochs=payload.epochs,
            batch_size=payload.batch_size,
            image_size=payload.image_size,
            seed=payload.seed,
            validation_split=payload.validation_split,
            learning_rate=payload.learning_rate,
            export_name=payload.export_name,
        )
        job_id = str(uuid.uuid4())
        return {"job_id": job_id, "status": "training", "model": config.model_name}
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while starting training")
        raise HTTPException(status_code=500, detail="Unexpected error while starting training.") from exc


@app.post("/api/convert-tfjs")
def convert_tfjs(model_name: str = Query(..., min_length=1)) -> dict[str, Any]:
    # Find the model path
    from pathlib import Path
    models_dir = resolve_models_dir()
    keras_path = models_dir / model_name / "best.keras"
    output_dir = Path("public/models") / model_name
    if not keras_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {keras_path}")
    job_id = str(uuid.uuid4())
    result = start_tfjs_conversion(str(keras_path), str(output_dir), job_id)
    return result

@app.get("/api/convert-tfjs/status")
def convert_tfjs_status() -> dict[str, Any]:
    return get_tfjs_state()


@app.post("/jira/sync", response_model=JiraSyncResponse)
def jira_sync(_payload: JiraSyncRequest) -> JiraSyncResponse:
    try:
        return jira_rag_service.sync()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while syncing Jira issues")
        raise HTTPException(status_code=500, detail="Unexpected error while syncing Jira issues.") from exc


@app.post("/jira/chat", response_model=JiraChatResponse)
def jira_chat(payload: JiraChatRequest) -> JiraChatResponse:
    try:
        return jira_rag_service.chat(payload.message, payload.top_k, payload.project_keys)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while chatting with Jira RAG")
        raise HTTPException(status_code=500, detail="Unexpected error while chatting with Jira RAG.") from exc


@app.get("/jira/projects", response_model=list[JiraProjectOption])
def jira_projects() -> list[JiraProjectOption]:
    try:
        return jira_rag_service.list_projects()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while listing Jira projects")
        raise HTTPException(status_code=500, detail="Unexpected error while listing Jira projects.") from exc


def main() -> None:
    import uvicorn

    host = os.getenv("VISION_AI_HOST", "127.0.0.1")
    port = int(os.getenv("VISION_AI_PORT", "8000"))
    uvicorn.run("backend.api:app", host=host, port=port, reload=True)


if __name__ == "__main__":
    main()
