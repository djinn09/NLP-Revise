"""Module defines data models for essay grading.

It includes:
- EssayScores: A model for storing essay scoring details.
- Essay: A model for representing essays and their reference texts.
"""

from __future__ import annotations

from pydantic import BaseModel


class EssayScores(BaseModel):
    """Data model for essay scores.

    Attributes:
        semantic_score (float | None): The semantic similarity score (None if not calculated).

    """

    semantic_score: float | None


class Essay(BaseModel):
    """Data model for essays.

    Attributes:
        text (str | list[str]): The essay text to be scored.
        reference (str | list[str]): The reference text(s) to compare against.

    """

    text: str | list[str]
    reference: str | list[str]
