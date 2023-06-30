import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseModel, BaseSettings


class RequestBody(BaseModel):
    text: str


class Settings(BaseSettings):
    app_name: str = "Coref API"
    app_version: str = "0.0.1"
    app_description: str = "Coreference resolution API"
    ALLEN_NLP_MODEL_URL = "models/coref-spanbert-large-2021.03.10.tar.gz"
    NEURALCOREF_CACHE = "models/neural_coref_models"
    DEBUG = os.getenv("DEBUG", True)
    PORT = os.getenv("PORT", 5000)
    HOST = os.getenv("HOST", "0.0.0.0")


class ResponseBody(BaseModel):
    error: Optional[str] = None
    text: Optional[str] = None
    neural_cluster: Optional[List[str]] = None
    coref_cluster: Optional[List[str]] = None


@lru_cache()
def get_settings():
    return Settings()


setting = get_settings()
