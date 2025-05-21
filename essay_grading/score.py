"""Module defines data models for essay grading.

It includes:
- EssayScores: A model for storing essay scoring details.
- score_essay: A function that scores essays against reference texts.
"""

from __future__ import annotations

from app_types import EssayScores, KeywordMatcherConfig
from key_word_match import SimilarityCalculator
from keyword_matcher import KeywordMatcher
from semantic_match import SemanticCosineSimilarity
from settings import semantic_model, settings, similarity_config # settings is imported here
from text_features import SinglePairAnalysisInput, run_single_pair_text_analysis


def score_essay(essay: str, reference: str) -> EssayScores: # Return type hint updated
    """Score an essay based on its semantic similarity to a reference text.

    Args:
        essay (str): The essay to be scored.
        reference (str): The reference text to compare against.

    Returns:
        EssayScores: An object containing all calculated scores including the final_score.

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
    semantic_score_result = sentence_semantic_model.calculate_similarity(
        essay,
        reference,
        metrics_to_calculate=["cosine"],
    )
    text_features_input = SinglePairAnalysisInput(
        model_answer=reference,
        student_text=essay,
    )
    individual_pair_results = run_single_pair_text_analysis(text_features_input)
    # Calculate the keyword similarity score
    config_pos = KeywordMatcherConfig(use_pos_tagging=True)
    keyword_matcher_obj = KeywordMatcher(config=config_pos) # Renamed to avoid conflict
    keyword_matcher_results = keyword_matcher_obj.find_matches_and_score(reference, essay)
    similarity_metrics = keyword_similarity_calculator.calculate_single_pair(reference, essay)
    
    current_semantic_score = None
    if semantic_score_result is not None:
        current_semantic_score = semantic_score_result.cosine

    scores_obj = EssayScores(
        semantic_score=current_semantic_score,
        similarity_metrics=similarity_metrics,
        text_score=individual_pair_results,
        keyword_matcher=keyword_matcher_results.keywords_matcher_result,
    )

    # Calculate final score
    s_score = scores_obj.semantic_score if scores_obj.semantic_score is not None else 0.0
    
    # Ensure keyword_matcher and similarity_metrics are not None before accessing their attributes
    k_score = 0.0
    if scores_obj.keyword_matcher: # This is KeywordMatcherScore type
        k_score = scores_obj.keyword_matcher.keyword_coverage_score
    
    t_score = 0.0
    if scores_obj.similarity_metrics: # This is SimilarityMetrics type
        t_score = scores_obj.similarity_metrics.tfidf_cosine_similarity if scores_obj.similarity_metrics.tfidf_cosine_similarity is not None else 0.0

    weights = settings.score_weights

    scores_obj.final_score = (s_score * weights.semantic_score_weight) + \
                             (k_score * weights.keyword_coverage_score_weight) + \
                             (t_score * weights.tfidf_cosine_similarity_weight)
    
    return scores_obj
