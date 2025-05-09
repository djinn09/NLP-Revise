"""Module defines data models for essay grading.

It includes:
- EssayScores: A model for storing essay scoring details.
- Essay: A model for representing essays and their reference texts.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Essay(BaseModel):
    """Data model for essays.

    Attributes:
        text (str | list[str]): The essay text to be scored.
        reference (str | list[str]): The reference text(s) to compare against.

    """

    text: str | list[str]
    reference: str | list[str]


class SimilarityMetrics(BaseModel):
    """Data model defining the structure for the dictionary of calculated similarity metrics.

    All fields are optional as some metrics might fail or not be applicable.
    """

    # Basic String / Sequence Metrics (often on lowercase strings)
    ratio: Optional[float] = Field(default=0.0, description="difflib.SequenceMatcher.ratio()")
    normalized_levenshtein: Optional[float] = Field(default=0.0, description="Normalized Levenshtein similarity.")
    jaro_winkler: Optional[float] = Field(default=0.0, description="Jaro-Winkler similarity.")
    metric_lcs_similarity: Optional[float] = Field(
        default=0.0,
        description="Similarity based on Longest Common Subsequence.",
    )
    qgram2_distance: Optional[float] = Field(
        default=0.0,
        description="Q-gram distance (n=2). Lower is more similar.",
    )
    qgram3_distance: Optional[float] = Field(
        default=0.0,
        description="Q-gram distance (n=3). Lower is more similar.",
    )
    qgram4_distance: Optional[float] = Field(
        default=0.0,
        description="Q-gram distance (n=4). Lower is more similar.",
    )

    cosine_char_2gram: Optional[float] = Field(
        default=0.0,
        description="Cosine similarity on character 2-grams.",
    )
    jaccard_char_2gram: Optional[float] = Field(
        default=0.0,
        description="Jaccard similarity on character 2-grams.",
    )

    # RapidFuzz Metrics (optimized fuzzy matching, scores typically 0-1 after /100.0)
    rfuzz_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz simple ratio.",
    )
    rfuzz_partial_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz partial ratio.",
    )
    rfuzz_token_set_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz token set ratio.",
    )
    rfuzz_token_sort_ratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz token sort ratio.",
    )
    rfuzz_wratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz weighted ratio (can exceed 1.0).",
    )
    rfuzz_qratio: Optional[float] = Field(
        default=0.0,
        description="RapidFuzz quick ratio.",
    )

    # FuzzyWuzzy Metrics (if available, scores typically 0-1 after /100.0)
    fz_uqratio: Optional[float] = Field(
        default=0.0,
        description="FuzzyWuzzy UQRatio (unicode quick ratio).",
    )
    fz_uwratio: Optional[float] = Field(
        default=0.0,
        description="FuzzyWuzzy UWRatio (unicode weighted ratio).",
    )

    # BLEU Score (translation quality metric, adapted for similarity)
    bleu_score: Optional[float] = Field(
        default=0.0,
        description="BLEU score, typically BLEU-4.",
    )

    # BM25 Score (ranking function, adapted for similarity)
    bm25: Optional[float] = Field(
        default=0.0,
        description="BM25 relevance score.",
    )

    # TF-IDF Based Metrics
    tfidf_cosine_similarity: Optional[float] = Field(
        default=0.0,
        description="Cosine similarity of TF-IDF vectors.",
    )
    tfidf_euclidean_distance: Optional[float] = Field(
        default=0.0,
        description="Euclidean distance of TF-IDF vectors.",
    )
    tfidf_manhattan_distance: Optional[float] = Field(
        default=0.0,
        description="Manhattan distance of TF-IDF vectors.",
    )
    tfidf_jaccard_distance: Optional[float] = Field(
        default=0.0,
        description="Jaccard distance of binarized TF-IDF vectors.",
    )
    tfidf_hamming_distance: Optional[float] = Field(
        default=0.0,
        description="Hamming distance of binarized TF-IDF vectors.",
    )
    tfidf_minkowski_distance: Optional[float] = Field(
        default=0.0,
        description="Minkowski distance of TF-IDF vectors.",
    )


class TfidfConfig(BaseModel):
    """Configuration for the scikit-learn TfidfVectorizer.

    Defines parameters like token pattern, n-gram range, and document frequency thresholds.
    """

    token_pattern: str = r"(?u)\b\w\w+\b"  # Default pattern to extract words of 2+ chars  # noqa: S105
    ngram_range: tuple[int, int] = Field(
        default=(1, 1),
        description="Range of n-grams, e.g., (1, 2) for unigrams and bigrams.",
    )
    max_df: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Ignore terms with document frequency higher than this threshold.",
    )
    min_df: int = Field(default=1, ge=0, description="Ignore terms with document frequency lower than this threshold.")


# --- Main SimilarityCalculator Class ---
class SimilarityCalculatorConfig(BaseModel):
    """Configuration for the main SimilarityCalculator.

    Allows setting preferences for lemmatization, stopwords, and TF-IDF parameters.
    """

    use_lemmatization: bool = Field(
        default=True,
        description="Enable/disable lemmatization.",
    )
    use_stopwords: bool = Field(
        default=True,
        description="Enable/disable stop-word removal.",
    )
    custom_stop_words: Optional[list[str]] = Field(
        default=None,
        description="list of custom stopwords to use or add.",
    )
    tfidf_config: TfidfConfig = Field(
        default_factory=TfidfConfig,
        description="Configuration for TF-IDF vectorization.",
    )
    # bleu_smoothing_function_name: Optional[str] = None # Example if passing smoothing by name



class EssayScores(BaseModel):
    """Data model for essay scores.

    Attributes:
        semantic_score (float | None): The semantic similarity score (None if not calculated).

    """

    semantic_score: float | None = 0.0
    similarity_metrics: SimilarityMetrics

