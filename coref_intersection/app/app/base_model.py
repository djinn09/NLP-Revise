"""Module defines the data models and settings for the Coreference Resolution API.

It includes:
- RequestBody: Model for the API request body.
- ResponseBody: Model for the API response body.
- Settings: Configuration settings for the API.
- get_settings: Function to retrieve cached settings.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, BaseSettings


class RequestBody(BaseModel):
    """Request body for the API."""

    text: str


class Settings(BaseSettings):
    """Settings for the API."""

    app_name: str = "Coref API"
    app_version: str = "0.0.1"
    app_description: str = "Coreference resolution API"
    ALLEN_NLP_MODEL_URL: str = "models/coref-spanbert-large-2021.03.10.tar.gz"
    NEURALCOREF_CACHE: str = "models/neural_coref_models"
    DEBUG: bool = os.getenv("DEBUG", "1") == "1"
    PORT: str = os.getenv("PORT", "5000")
    HOST = os.getenv("HOST", "127.0.0.1")


class ResponseBody(BaseModel):
    """Response body for the API."""

    error: Optional[str] = None
    text: Optional[str] = None
    neural_cluster: Optional[List[str]] = None
    coref_cluster: Optional[List[str]] = None


@lru_cache()
def get_settings() -> Settings:
    """Return an instance of Settings with default values.

    The @lru_cache() decorator caches the result of this function, so the same
    Settings instance is returned every time this function is called.
    """
    return Settings()


setting = get_settings()
