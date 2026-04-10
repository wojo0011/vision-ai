from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

from .config import JiraSettings
from .embeddings import load_embedding_model
from .schemas import JiraChunkRecord

logger = logging.getLogger(__name__)


class JiraVectorStore:
    def __init__(self, settings: JiraSettings) -> None:
        self.settings = settings
        self._index: faiss.Index | None = None
        self._chunks: list[JiraChunkRecord] = []

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = load_embedding_model(self.settings.embedding_model_name)
        embeddings = model.encode(
            texts,
            batch_size=self.settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype("float32")

    def rebuild(self, chunks: list[JiraChunkRecord]) -> None:
        self.settings.ensure_directories()
        self._chunks = chunks

        with self.settings.chunks_path.open("w", encoding="utf-8") as handle:
            json.dump([chunk.model_dump() for chunk in chunks], handle, ensure_ascii=False, indent=2)

        if not chunks:
            self._index = None
            if self.settings.faiss_index_path.exists():
                self.settings.faiss_index_path.unlink()
            return

        embeddings = self._encode([chunk.text for chunk in chunks])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(self.settings.faiss_index_path))

        with self.settings.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "embedding_model": self.settings.embedding_model_name,
                    "chunk_count": len(chunks),
                },
                handle,
                indent=2,
            )

        self._index = index
        logger.info("Rebuilt Jira FAISS index with %s chunks", len(chunks))

    def load(self) -> bool:
        if not self.settings.chunks_path.exists() or not self.settings.faiss_index_path.exists():
            return False

        with self.settings.chunks_path.open("r", encoding="utf-8") as handle:
            raw_chunks = json.load(handle)

        self._chunks = [JiraChunkRecord.model_validate(item) for item in raw_chunks]
        self._index = faiss.read_index(str(self.settings.faiss_index_path))
        return True

    def search(self, query: str, top_k: int, project_keys: list[str] | None = None) -> list[tuple[JiraChunkRecord, float]]:
        if self._index is None and not self.load():
            return []

        if self._index is None or not self._chunks:
            return []

        query_vector = self._encode([query])
        search_k = min(max(top_k * 5, top_k), len(self._chunks))
        scores, indices = self._index.search(query_vector, search_k)
        allowed_projects = {key for key in (project_keys or []) if key}

        results: list[tuple[JiraChunkRecord, float]] = []
        for row_index, score in zip(indices[0], scores[0], strict=False):
            if row_index < 0 or row_index >= len(self._chunks):
                continue
            chunk = self._chunks[row_index]
            if allowed_projects and chunk.metadata.get("project_key") not in allowed_projects:
                continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)
