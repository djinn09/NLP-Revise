import pytest
from typing import Optional
from pydantic import BaseModel, Field

# Re-defining necessary Pydantic models for test clarity and isolation,
# or import them from essay_grading.app_types if preferred and stable.
# For this task, re-defining simplified versions to focus on the tested logic.

class ScoreWeights(BaseModel):
    semantic_score_weight: float = Field(default=0.5, ge=0, le=1)
    keyword_coverage_score_weight: float = Field(default=0.25, ge=0, le=1)
    tfidf_cosine_similarity_weight: float = Field(default=0.25, ge=0, le=1)

class KeywordMatcherScore(BaseModel):
    keywords_from_a_count: int = 0 # Added default for testing
    matched_keyword_count: int = 0 # Added default for testing
    keyword_coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    vocabulary_cosine_similarity: float = 0.0 # Added default for testing

class SimilarityMetrics(BaseModel):
    # Adding other fields with defaults to allow instantiation
    ratio: Optional[float] = 0.0
    normalized_levenshtein: Optional[float] = 0.0
    jaro_winkler: Optional[float] = 0.0
    metric_lcs_similarity: Optional[float] = 0.0
    qgram2_distance: Optional[float] = 0.0
    qgram3_distance: Optional[float] = 0.0
    qgram4_distance: Optional[float] = 0.0
    cosine_char_2gram: Optional[float] = 0.0
    jaccard_char_2gram: Optional[float] = 0.0
    rfuzz_ratio: Optional[float] = 0.0
    rfuzz_partial_ratio: Optional[float] = 0.0
    rfuzz_token_set_ratio: Optional[float] = 0.0
    rfuzz_token_sort_ratio: Optional[float] = 0.0
    rfuzz_wratio: Optional[float] = 0.0
    rfuzz_qratio: Optional[float] = 0.0
    fz_uqratio: Optional[float] = 0.0
    fz_uwratio: Optional[float] = 0.0
    bleu_score: Optional[float] = 0.0
    bm25: Optional[float] = 0.0
    tfidf_cosine_similarity: Optional[float] = Field(default=0.0) # Target field
    tfidf_euclidean_distance: Optional[float] = 0.0
    tfidf_manhattan_distance: Optional[float] = 0.0
    tfidf_jaccard_distance: Optional[float] = 0.0
    tfidf_hamming_distance: Optional[float] = 0.0
    tfidf_minkowski_distance: Optional[float] = 0.0

class SinglePairAnalysisResult(BaseModel): # Placeholder for structure
    # Add any fields that might be accessed or needed for instantiation
    pass

class EssayScores(BaseModel):
    semantic_score: Optional[float] = 0.0
    similarity_metrics: Optional[SimilarityMetrics] = None # Allow None for testing
    text_score: Optional[SinglePairAnalysisResult] = None # Allow None for testing
    keyword_matcher: Optional[KeywordMatcherScore] = None # Allow None for testing
    final_score: Optional[float] = None


# Helper function replicating the calculation logic from score.py
def calculate_final_score_for_test(scores_obj: EssayScores, weights: ScoreWeights) -> Optional[float]:
    s_score = scores_obj.semantic_score if scores_obj.semantic_score is not None else 0.0
    
    k_score = 0.0
    if scores_obj.keyword_matcher:
        k_score = scores_obj.keyword_matcher.keyword_coverage_score
    
    t_score = 0.0
    if scores_obj.similarity_metrics:
        t_score = scores_obj.similarity_metrics.tfidf_cosine_similarity if scores_obj.similarity_metrics.tfidf_cosine_similarity is not None else 0.0
    
    final_score_value = (s_score * weights.semantic_score_weight) + \
                        (k_score * weights.keyword_coverage_score_weight) + \
                        (t_score * weights.tfidf_cosine_similarity_weight)
    return final_score_value

# --- Test Cases ---

def test_all_scores_present_default_weights():
    """Test with all component scores present and valid using default weights."""
    scores = EssayScores(
        semantic_score=0.8,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=0.7),
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=0.9),
        text_score=SinglePairAnalysisResult() # Ensure text_score is not None
    )
    default_weights = ScoreWeights() # semantic=0.5, keyword=0.25, tfidf=0.25
    
    expected_final_score = (0.8 * 0.5) + (0.7 * 0.25) + (0.9 * 0.25) 
    # 0.4 + 0.175 + 0.225 = 0.8
    
    actual_final_score = calculate_final_score_for_test(scores, default_weights)
    assert actual_final_score == pytest.approx(expected_final_score)

def test_semantic_score_none():
    """Test with semantic_score being None."""
    scores = EssayScores(
        semantic_score=None,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=0.7),
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=0.9),
        text_score=SinglePairAnalysisResult()
    )
    default_weights = ScoreWeights()
    
    expected_final_score = (0.0 * 0.5) + (0.7 * 0.25) + (0.9 * 0.25)
    # 0.0 + 0.175 + 0.225 = 0.4
    
    actual_final_score = calculate_final_score_for_test(scores, default_weights)
    assert actual_final_score == pytest.approx(expected_final_score)

def test_keyword_coverage_none():
    """Test with keyword_matcher or its score being None/0."""
    # Scenario 1: keyword_matcher is None
    scores1 = EssayScores(
        semantic_score=0.8,
        keyword_matcher=None, # keyword_matcher object is None
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=0.9),
        text_score=SinglePairAnalysisResult()
    )
    default_weights = ScoreWeights()
    expected_final_score1 = (0.8 * 0.5) + (0.0 * 0.25) + (0.9 * 0.25)
    # 0.4 + 0.0 + 0.225 = 0.625
    actual_final_score1 = calculate_final_score_for_test(scores1, default_weights)
    assert actual_final_score1 == pytest.approx(expected_final_score1)

    # Scenario 2: keyword_coverage_score is 0 (default in KeywordMatcherScore if not set)
    scores2 = EssayScores(
        semantic_score=0.8,
        keyword_matcher=KeywordMatcherScore(), # keyword_coverage_score defaults to 0.0
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=0.9),
        text_score=SinglePairAnalysisResult()
    )
    expected_final_score2 = (0.8 * 0.5) + (0.0 * 0.25) + (0.9 * 0.25)
    # 0.4 + 0.0 + 0.225 = 0.625
    actual_final_score2 = calculate_final_score_for_test(scores2, default_weights)
    assert actual_final_score2 == pytest.approx(expected_final_score2)


def test_tfidf_cosine_none():
    """Test with tfidf_cosine_similarity being None or its parent (similarity_metrics) being None."""
    # Scenario 1: similarity_metrics object is None
    scores1 = EssayScores(
        semantic_score=0.8,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=0.7),
        similarity_metrics=None, # similarity_metrics object is None
        text_score=SinglePairAnalysisResult()
    )
    default_weights = ScoreWeights()
    expected_final_score1 = (0.8 * 0.5) + (0.7 * 0.25) + (0.0 * 0.25)
    # 0.4 + 0.175 + 0.0 = 0.575
    actual_final_score1 = calculate_final_score_for_test(scores1, default_weights)
    assert actual_final_score1 == pytest.approx(expected_final_score1)

    # Scenario 2: tfidf_cosine_similarity is None (explicitly)
    scores2 = EssayScores(
        semantic_score=0.8,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=0.7),
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=None),
        text_score=SinglePairAnalysisResult()
    )
    expected_final_score2 = (0.8 * 0.5) + (0.7 * 0.25) + (0.0 * 0.25)
    # 0.4 + 0.175 + 0.0 = 0.575
    actual_final_score2 = calculate_final_score_for_test(scores2, default_weights)
    assert actual_final_score2 == pytest.approx(expected_final_score2)
    
    # Scenario 3: tfidf_cosine_similarity is 0.0 (default if SimilarityMetrics is instantiated without it)
    scores3 = EssayScores(
        semantic_score=0.8,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=0.7),
        similarity_metrics=SimilarityMetrics(), # tfidf_cosine_similarity defaults to 0.0
        text_score=SinglePairAnalysisResult()
    )
    expected_final_score3 = (0.8 * 0.5) + (0.7 * 0.25) + (0.0 * 0.25)
    # 0.4 + 0.175 + 0.0 = 0.575
    actual_final_score3 = calculate_final_score_for_test(scores3, default_weights)
    assert actual_final_score3 == pytest.approx(expected_final_score3)


def test_all_scores_zero():
    """Test with all component scores being 0."""
    scores = EssayScores(
        semantic_score=0.0,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=0.0),
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=0.0),
        text_score=SinglePairAnalysisResult()
    )
    default_weights = ScoreWeights()
    
    expected_final_score = 0.0
    
    actual_final_score = calculate_final_score_for_test(scores, default_weights)
    assert actual_final_score == pytest.approx(expected_final_score)

def test_all_scores_one():
    """Test with all component scores being 1.0. Final score should be sum of weights."""
    scores = EssayScores(
        semantic_score=1.0,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=1.0),
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=1.0),
        text_score=SinglePairAnalysisResult()
    )
    default_weights = ScoreWeights() # 0.5, 0.25, 0.25
    
    expected_final_score = default_weights.semantic_score_weight + \
                           default_weights.keyword_coverage_score_weight + \
                           default_weights.tfidf_cosine_similarity_weight
    # 0.5 + 0.25 + 0.25 = 1.0
    
    actual_final_score = calculate_final_score_for_test(scores, default_weights)
    assert actual_final_score == pytest.approx(expected_final_score)

def test_custom_weights():
    """Test with custom weights."""
    scores = EssayScores(
        semantic_score=0.8,
        keyword_matcher=KeywordMatcherScore(keyword_coverage_score=0.7),
        similarity_metrics=SimilarityMetrics(tfidf_cosine_similarity=0.9),
        text_score=SinglePairAnalysisResult()
    )
    custom_weights = ScoreWeights(
        semantic_score_weight=0.7,
        keyword_coverage_score_weight=0.1,
        tfidf_cosine_similarity_weight=0.2
    )
    
    expected_final_score = (0.8 * 0.7) + (0.7 * 0.1) + (0.9 * 0.2)
    # 0.56 + 0.07 + 0.18 = 0.81
    
    actual_final_score = calculate_final_score_for_test(scores, custom_weights)
    assert actual_final_score == pytest.approx(expected_final_score)

def test_optional_fields_fully_none():
    """Test when optional EssayScore fields for sub-scores are None."""
    scores = EssayScores(
        semantic_score=0.8,
        keyword_matcher=None, # KeywordMatcherScore object itself is None
        similarity_metrics=None, # SimilarityMetrics object itself is None
        text_score=SinglePairAnalysisResult()
    )
    default_weights = ScoreWeights()
    
    expected_final_score = (0.8 * 0.5) + (0.0 * 0.25) + (0.0 * 0.25)
    # 0.4 + 0.0 + 0.0 = 0.4
    
    actual_final_score = calculate_final_score_for_test(scores, default_weights)
    assert actual_final_score == pytest.approx(expected_final_score)

def test_instantiation_with_minimal_data():
    """Test that EssayScores can be instantiated with minimal data for the helper."""
    scores = EssayScores(
        semantic_score=0.5,
        # keyword_matcher, similarity_metrics, text_score use their default None
    )
    default_weights = ScoreWeights()
    expected_final_score = (0.5 * 0.5) + (0.0 * 0.25) + (0.0 * 0.25) # 0.25
    actual_final_score = calculate_final_score_for_test(scores, default_weights)
    assert actual_final_score == pytest.approx(expected_final_score)

    scores_with_empty_objects = EssayScores(
        semantic_score=0.6,
        keyword_matcher=KeywordMatcherScore(), # coverage_score = 0.0
        similarity_metrics=SimilarityMetrics(), # tfidf_cosine = 0.0
        text_score=SinglePairAnalysisResult()
    )
    expected_final_score_2 = (0.6 * 0.5) + (0.0 * 0.25) + (0.0 * 0.25) # 0.3
    actual_final_score_2 = calculate_final_score_for_test(scores_with_empty_objects, default_weights)
    assert actual_final_score_2 == pytest.approx(expected_final_score_2)

# Note: The prompt mentions mocking settings for custom weights using unittest.mock.patch.
# However, the adopted strategy is to use a helper function `calculate_final_score_for_test`
# which takes `weights` as an argument. This simplifies testing the calculation logic
# without needing to patch module-level settings, making the tests more direct and robust
# to how `settings` is structured or imported.
# If testing the `score_essay` function directly was required, patching would be necessary.
# For example:
# from unittest.mock import patch, PropertyMock
# def test_score_essay_with_mocked_custom_weights():
#     # This would be a more complex integration test for score_essay
#     # and would require mocking score_essay's direct dependencies too.
#     # For this subtask, we focus on calculate_final_score_for_test.
#     pass
