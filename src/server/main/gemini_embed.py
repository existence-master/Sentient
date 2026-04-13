"""Gemini text embeddings via the ``google-genai`` SDK.

Replaces the deprecated ``google.generativeai`` package; see:
https://github.com/google-gemini/deprecated-generative-ai-python
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from google import genai
from google.genai import types

from main.config import EMBEDDING_MODEL_NAME, GEMINI_API_KEY

_client: Optional[genai.Client] = None
_resolved_model_name: Optional[str] = None


def _client_instance() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def embedding_model_id() -> str:
    """Model id for embed_content (lazy, validates API key)."""
    global _resolved_model_name
    if _resolved_model_name is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set.")
        _resolved_model_name = EMBEDDING_MODEL_NAME
    return _resolved_model_name


def ensure_embedding_ready() -> None:
    """Validate config and warm the client; safe to call from app startup."""
    _client_instance()
    embedding_model_id()


def get_normalized_embedding(text: str, task_type: str) -> np.ndarray:
    """
    Single-text embedding with L2 normalization (768 dimensions).

    Task types match the Gemini API: RETRIEVAL_QUERY, RETRIEVAL_DOCUMENT, etc.
    """
    response = _client_instance().models.embed_content(
        model=embedding_model_id(),
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )
    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("Gemini embed_content returned no embedding values.")
    embedding_np = np.array(response.embeddings[0].values, dtype=np.float32)
    norm = float(np.linalg.norm(embedding_np))
    if norm == 0:
        return embedding_np
    return embedding_np / norm
