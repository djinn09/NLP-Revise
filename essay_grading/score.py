"""Module defines data models for essay grading.

It includes:
- EssayScores: A model for storing essay scoring details.
- score_essay: A function that scores essays against reference texts.
"""

from __future__ import annotations

from app_types import EssayScores
from key_word_match import SimilarityCalculator
from semantic_match import SemanticCosineSimilarity
from settings import semantic_model, settings, similarity_config
from text_features import SinglePairAnalysisInput, run_single_pair_text_analysis


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
    # Initialize the keyword similarity calculator
    keyword_similarity_calculator = SimilarityCalculator(
        config=similarity_config,
    )
    # Calculate the semantic similarity score
    semantic_score = sentence_semantic_model.calculate_similarity(
        essay,
        reference,
        metrics_to_calculate=["cosine"],
    )
    text_features_input = SinglePairAnalysisInput(
        model_answer=reference,
        student_text=essay,
    )
    individual_pair_results = run_single_pair_text_analysis(text_features_input)

    if individual_pair_results.graph_similarity:
        print(f"Graph Similarity: {individual_pair_results.graph_similarity.similarity_score:.4f}")
    if individual_pair_results.plagiarism_score:
        print(f"Fast Plagiarism Overlap: {individual_pair_results.plagiarism_score.overlap_percentage:.2%}")
    if individual_pair_results.overlap_coefficient:
        print(f"Overlap Coefficient: {individual_pair_results.overlap_coefficient.coefficient:.4f}")
    if individual_pair_results.dice_coefficient:
        print(f"Sørensen-Dice Coefficient: {individual_pair_results.dice_coefficient.coefficient:.4f}")
    if individual_pair_results.char_equality_score:
        print(f"Character by Character Equality: {individual_pair_results.char_equality_score.score}")
    similarity_metrics = keyword_similarity_calculator.calculate_single_pair(reference, essay)
    if semantic_score is not None:
        return EssayScores(
            semantic_score=semantic_score.cosine,
            similarity_metrics=similarity_metrics,
            text_score=individual_pair_results,
        )
    return EssayScores(
        similarity_metrics=similarity_metrics,
        text_score=individual_pair_results,
    )
