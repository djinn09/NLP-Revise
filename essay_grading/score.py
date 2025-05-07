from __future__ import annotations

from app_types import EssayScores
from semantic_match import SemanticCosineSimilarity
from settings import semantic_model, settings


def score_essay(essay: str, reference: str) -> EssayScores:
    """Score an essay based on its semantic similarity to a reference text.

    Args:
        essay (str): The essay to be scored.
        reference (str): The reference text to compare against.

    Returns:
        float: A score between 0 and 1 representing the semantic similarity.

    """
    # Initialize the semantic cosine similarity model
    sentence_semantic_model = SemanticCosineSimilarity(
        model=semantic_model,
        chunk_size=settings.semantic.chunk_size,
        batch_size=settings.semantic.batch_size,
    )

    # Calculate the semantic similarity score
    semantic_score = sentence_semantic_model.calculate_similarity(essay, reference)

    return EssayScores(semantic_score=semantic_score)
