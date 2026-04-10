from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def load_embedding_model(model_name: str) -> SentenceTransformer:
    logger.info("Loading sentence transformer model %s", model_name)
    return SentenceTransformer(model_name)

