# Use annotations for cleaner type hinting (requires Python 3.7+)
from __future__ import annotations

import logging
import math  # For isnan if needed, though not directly used with StandardScaler
from typing import Dict, List, Optional, Union

import numpy as np
import textstat
from numpy.linalg import norm as np_norm  # Alias to avoid conflict if norm is used elsewhere
from pydantic import BaseModel, Field, field_validator
from sklearn.preprocessing import StandardScaler

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Pydantic Models ---


class BaseReadabilityMetrics(BaseModel):
    """
    Base model for readability metrics, defining common fields and validators.
    This can be inherited by Raw and Normalized metric models.
    """

    flesch_reading_ease: float = Field(..., description="Flesch Reading Ease score. Higher is easier.")
    flesch_kincaid_grade: float = Field(..., description="Flesch-Kincaid Grade Level.")
    smog_index: float = Field(..., description="SMOG Index. Estimates years of education needed.")
    gunning_fog: float = Field(..., description="Gunning Fog Index.")
    dale_chall: float = Field(..., description="Dale-Chall Readability Score.")
    automated_readability_index: float = Field(..., description="Automated Readability Index (ARI).")
    coleman_liau_index: float = Field(..., description="Coleman-Liau Index.")
    linsear_write_formula: float = Field(..., description="Linsear Write Formula.")
    difficult_words: int = Field(..., ge=0, description="Count of words considered difficult.")
    sentence_count: int = Field(..., ge=0, description="Total number of sentences.")
    avg_sentence_length: float = Field(..., ge=0.0, description="Average words per sentence.")
    syllable_count: int = Field(..., ge=0, description="Total number of syllables.")
    lexicon_count: int = Field(..., ge=0, description="Total words (punctuation removed).")
    # text_standard is removed as it's not in your get_readability_metrics function

    @field_validator(
        "difficult_words",
        "sentence_count",
        "syllable_count",
        "lexicon_count",
        mode="before",  # Validate before Pydantic's type conversion
    )
    @classmethod
    def ensure_integer_counts(cls, value: Union[int, float]) -> int:
        """Ensures count metrics are integers. Textstat might return floats."""
        if isinstance(value, float):
            # Round and convert to int; useful if textstat sometimes gives float for counts
            return int(round(value))
        if not isinstance(value, int):
            # This case should ideally not be hit if input is from textstat and then rounded
            raise ValueError(f"Count metric must be an integer, got {type(value)}")
        return value

    @field_validator(
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "smog_index",
        "gunning_fog",
        "dale_chall",
        "automated_readability_index",
        "coleman_liau_index",
        "linsear_write_formula",
        "avg_sentence_length",
        mode="before",
    )
    @classmethod
    def ensure_float_scores(cls, value: Union[int, float]) -> float:
        """Ensures score metrics are floats."""
        if not isinstance(value, (int, float)):
            raise ValueError(f"Score metric must be a number, got {type(value)}")
        return float(value)


class ReadabilityMetricsRaw(BaseReadabilityMetrics):
    """
    Data model for storing raw readability metrics calculated for a given text.
    Inherits fields and validators from BaseReadabilityMetrics.
    """

    pass  # No additional fields or validators needed beyond the base


class ReadabilityMetricsNormalized(BaseReadabilityMetrics):
    """
    Data model for storing normalized (e.g., standardized) readability metrics.
    The values will be scaled, but the structure matches the raw metrics.
    Inherits fields and validators from BaseReadabilityMetrics.
    Note: ge=0 validation for counts might not apply strictly after normalization
    if normalization can result in negative values (e.g. standard scaler).
    We can override or remove those specific field constraints for normalized data if needed.
    For simplicity here, we keep them, assuming normalization mostly scales around 0.
    """

    # If normalization can make counts negative, override Field definitions:
    # difficult_words: float = Field(..., description="Normalized count of difficult words.")
    # sentence_count: float = Field(..., description="Normalized count of sentences.")
    # ... and so on for other counts, changing type to float and removing ge=0 if necessary.
    # For StandardScaler, outputs are floats.
    difficult_words: float = Field(..., description="Normalized count of difficult words.")
    sentence_count: float = Field(..., description="Normalized count of sentences.")
    avg_sentence_length: float = Field(..., description="Normalized average words per sentence.")
    syllable_count: float = Field(..., description="Normalized total number of syllables.")
    lexicon_count: float = Field(..., description="Normalized total words (punctuation removed).")


class MetricDifferencesRaw(BaseModel):
    """
    Data model for storing the absolute differences between two sets of raw readability metrics.
    All difference values are non-negative floats.
    """

    flesch_reading_ease: float = Field(..., ge=0.0)
    flesch_kincaid_grade: float = Field(..., ge=0.0)
    smog_index: float = Field(..., ge=0.0)
    gunning_fog: float = Field(..., ge=0.0)
    dale_chall: float = Field(..., ge=0.0)
    automated_readability_index: float = Field(..., ge=0.0)
    coleman_liau_index: float = Field(..., ge=0.0)
    linsear_write_formula: float = Field(..., ge=0.0)
    difficult_words: float = Field(..., ge=0.0)
    sentence_count: float = Field(..., ge=0.0)
    avg_sentence_length: float = Field(..., ge=0.0)
    syllable_count: float = Field(..., ge=0.0)
    lexicon_count: float = Field(..., ge=0.0)


# --- Functions ---


def get_readability_metrics(text: str) -> ReadabilityMetricsRaw:
    """
    Computes a set of raw readability metrics for a given text using the `textstat` library.

    Args:
        text: The input string. An empty string or string with insufficient content
              might lead to errors or default/zero values from `textstat` functions.

    Returns:
        A ReadabilityMetricsRaw Pydantic model instance containing the calculated scores.

    Raises:
        TypeError: If the input `text` is not a string.
        Exception: Propagates exceptions from `textstat` if critical errors occur
                   (e.g., due to unsupported language if `textstat.set_lang` was used incorrectly,
                   or internal errors for highly unusual inputs).
    """
    if not isinstance(text, str):
        logger.error("Input 'text' for readability metrics must be a string.")
        raise TypeError("Input 'text' must be a string.")

    if not text.strip():
        logger.warning("Input text is empty or whitespace only. Readability metrics will likely be defaults or zeros.")
        # Provide default zero/neutral values for empty text to ensure consistent model structure.
        # This makes the function more robust against empty inputs.
        return ReadabilityMetricsRaw(
            flesch_reading_ease=0.0,  # Neutral or error value for ease with empty text
            flesch_kincaid_grade=0.0,
            smog_index=0.0,  # SMOG typically requires >= 30 sentences
            gunning_fog=0.0,
            dale_chall=0.0,
            automated_readability_index=0.0,
            coleman_liau_index=0.0,
            linsear_write_formula=0.0,
            difficult_words=0,
            sentence_count=0,
            avg_sentence_length=0.0,
            syllable_count=0,
            lexicon_count=0,
        )
    try:
        # Calculate all metrics using textstat
        raw_metrics_dict = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),  # type: ignore  # noqa: PGH003
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),  # type: ignore # noqa: PGH003
            "smog_index": textstat.smog_index(text),  # type: ignore  # noqa: PGH003
            "gunning_fog": textstat.gunning_fog(text),  # type: ignore  # noqa: PGH003
            "dale_chall": textstat.dale_chall_readability_score(text),  # type: ignore  # noqa: PGH003
            "automated_readability_index": textstat.automated_readability_index(text),  # type: ignore  # noqa: PGH003
            "coleman_liau_index": textstat.coleman_liau_index(text),  # type: ignore  # noqa: PGH003
            "linsear_write_formula": textstat.linsear_write_formula(text),  # type: ignore  # noqa: PGH003
            "difficult_words": textstat.difficult_words(text),  # type: ignore  # noqa: PGH003
            "sentence_count": textstat.sentence_count(text),  # type: ignore  # noqa: PGH003
            "avg_sentence_length": textstat.avg_sentence_length(text),  # type: ignore  # noqa: PGH003
            "syllable_count": textstat.syllable_count(text),  # type: ignore # noqa: PGH003
            "lexicon_count": textstat.lexicon_count(text, removepunct=True),  # type: ignore  # noqa: PGH003
        }
        # Instantiate and validate using the Pydantic model
        return ReadabilityMetricsRaw(**raw_metrics_dict)
    except Exception as e:
        logger.error(f"Error calculating readability metrics for text: '{text[:50]}...'. Error: {e}", exc_info=True)
        # Re-raise the exception to signal failure to the caller
        raise


def normalize_metrics(metrics_list: List[ReadabilityMetricsRaw]) -> List[ReadabilityMetricsNormalized]:
    """
    Standardizes each readability feature across a list of texts (corpus)
    to have zero mean and unit variance using `sklearn.preprocessing.StandardScaler`.

    Args:
        metrics_list: A list of ReadabilityMetricsRaw Pydantic model instances,
                      where each instance represents the raw metrics for one text.

    Returns:
        A list of ReadabilityMetricsNormalized Pydantic model instances,
        with metrics scaled. Returns an empty list if the input is empty.
        If input list contains only one item, StandardScaler will result in all zeros.
    """
    if not metrics_list:
        logger.warning("normalize_metrics received an empty list. Returning empty list.")
        return []
    if not all(isinstance(m, ReadabilityMetricsRaw) for m in metrics_list):
        raise TypeError("All items in metrics_list must be ReadabilityMetricsRaw instances.")

    # Use the defined fields from the Pydantic model to ensure consistent order
    # Pydantic model_fields is an ordered dictionary in Python 3.7+
    keys = list(ReadabilityMetricsRaw.model_fields.keys())

    # Convert list of Pydantic models to a NumPy array
    # Each row is a text, each column is a metric
    try:
        mat = np.array(
            [[getattr(m, key) for key in keys] for m in metrics_list],
            dtype=float,  # Ensure float for scaler
        )
    except Exception as e:
        logger.error(f"Error converting metrics list to NumPy array: {e}", exc_info=True)
        raise ValueError("Could not convert metrics list to NumPy array. Check metric values.") from e

    if mat.shape[0] < 1:  # Should be caught by `if not metrics_list:`
        return []

    # Initialize and apply StandardScaler
    scaler = StandardScaler()
    # Handle case with only one sample: fit_transform would result in NaNs/zeros
    # and scaler.scale_ would be zero, leading to division by zero if not handled.
    # StandardScaler's fit_transform on a single sample results in an array of zeros.
    if mat.shape[0] == 1:
        logger.warning(
            "Normalizing metrics with only one sample. "
            "StandardScaler will produce all zeros for the normalized metrics."
        )
        mat_norm = np.zeros_like(mat, dtype=float)
    else:
        try:
            mat_norm = scaler.fit_transform(mat)
        except ValueError as e_scale:  # e.g. if a column has zero variance
            logger.error(
                f"Error during StandardScaler fit_transform: {e_scale}. This can happen if a feature has zero variance (all values are the same).",
                exc_info=True,
            )
            # Fallback: return original values as "normalized" if scaling fails for such reasons
            # Or, more robustly, identify constant columns and set them to 0 in normalized output.
            # For now, let's raise to make the user aware.
            raise ValueError(
                f"StandardScaler failed, possibly due to zero variance in a feature: {e_scale}"
            ) from e_scale

    # Convert normalized NumPy array back to a list of Pydantic models
    normalized_metrics_obj_list: List[ReadabilityMetricsNormalized] = []
    for row in mat_norm:
        # Pydantic will convert float values from mat_norm to int for count fields in Normalized model
        # if those fields were still typed as int. Our Normalized model now uses float for counts.
        norm_dict = dict(zip(keys, row))
        try:
            normalized_metrics_obj_list.append(ReadabilityMetricsNormalized(**norm_dict))
        except Exception as e_pydantic:  # Catch Pydantic validation errors during reconstruction
            logger.error(
                f"Error creating ReadabilityMetricsNormalized model from normalized data: {norm_dict}. Error: {e_pydantic}",
                exc_info=True,
            )
            # Skip this problematic entry or raise
            raise ValueError("Failed to reconstruct Pydantic model for normalized metrics.") from e_pydantic

    return normalized_metrics_obj_list


def calculate_euclidean_distance(
    metrics1: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
    metrics2: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
) -> float:
    """
    Computes the Euclidean (L2) distance between two feature vectors
    represented by ReadabilityMetrics Pydantic models.

    Args:
        metrics1: Readability metrics for the first text.
        metrics2: Readability metrics for the second text.

    Returns:
        The Euclidean distance as a float.
    """
    if not isinstance(metrics1, BaseReadabilityMetrics) or not isinstance(metrics2, BaseReadabilityMetrics):
        raise TypeError("Inputs must be instances of a BaseReadabilityMetrics model.")

    # Extract values in a consistent order using model_fields
    keys = list(BaseReadabilityMetrics.model_fields.keys())
    arr1 = np.array([getattr(metrics1, key) for key in keys], dtype=float)
    arr2 = np.array([getattr(metrics2, key) for key in keys], dtype=float)

    distance = np_norm(arr1 - arr2)  # np.linalg.norm computes L2 norm by default
    return float(distance)


def calculate_manhattan_distance(
    metrics1: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
    metrics2: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
) -> float:
    """
    Computes the Manhattan (L1) distance between two feature vectors
    represented by ReadabilityMetrics Pydantic models.

    Args:
        metrics1: Readability metrics for the first text.
        metrics2: Readability metrics for the second text.

    Returns:
        The Manhattan distance as a float.
    """
    if not isinstance(metrics1, BaseReadabilityMetrics) or not isinstance(metrics2, BaseReadabilityMetrics):
        raise TypeError("Inputs must be instances of a BaseReadabilityMetrics model.")

    keys = list(BaseReadabilityMetrics.model_fields.keys())
    arr1 = np.array([getattr(metrics1, key) for key in keys], dtype=float)
    arr2 = np.array([getattr(metrics2, key) for key in keys], dtype=float)

    distance = np.abs(arr1 - arr2).sum()
    return float(distance)


def compare_raw_metrics_absolute_diff(
    metrics1: ReadabilityMetricsRaw, metrics2: ReadabilityMetricsRaw
) -> MetricDifferencesRaw:
    """
    Computes the absolute differences between raw scores of two ReadabilityMetricsRaw objects.
    This is useful for direct interpretability of score differences.

    Args:
        metrics1: Raw readability metrics for the first text.
        metrics2: Raw readability metrics for the second text.

    Returns:
        A MetricDifferencesRaw Pydantic model instance containing the absolute differences.
    """
    if not isinstance(metrics1, ReadabilityMetricsRaw) or not isinstance(metrics2, ReadabilityMetricsRaw):
        raise TypeError("Inputs must be ReadabilityMetricsRaw instances for raw comparison.")

    # Use model_dump for easy dict access, Pydantic models guarantee field presence
    m1_dict = metrics1.model_dump()
    m2_dict = metrics2.model_dump()

    differences_dict = {
        key: abs(m1_dict[key] - m2_dict[key])
        for key in m1_dict  # Assuming keys are identical due to common Pydantic model
    }
    return MetricDifferencesRaw(**differences_dict)


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting readability metrics processing example.")

    # Example texts: student vs. model essay
    student_text = "Education is the passport to the future, for tomorrow belongs to those who prepare for it today. The journey of learning is lifelong."
    model_text = "The future belongs to those who prepare for it today; education is their passport. Learning is a continuous voyage."
    another_text = "This is a very simple and short text. It has few words. Reading is easy."

    # --- Compute Raw Metrics ---
    try:
        logger.info("Computing raw metrics for student text...")
        student_metrics_raw = get_readability_metrics(student_text)
        logger.info("Computing raw metrics for model text...")
        model_metrics_raw = get_readability_metrics(model_text)
        logger.info("Computing raw metrics for another text...")
        another_metrics_raw = get_readability_metrics(another_text)
    except Exception as e_raw_metrics:
        logger.error(f"Failed to compute raw readability metrics: {e_raw_metrics}", exc_info=True)
        exit(1)  # Exit if essential data cannot be computed

    # --- Show Raw Differences (for interpretability) ---
    logger.info("Comparing raw metrics (student vs. model)...")
    raw_differences = compare_raw_metrics_absolute_diff(student_metrics_raw, model_metrics_raw)

    # --- Normalize Metrics Across a "Corpus" (here, just our example texts) ---
    # In a real scenario, this corpus would be much larger (e.g., all student essays).
    corpus_raw_metrics: List[ReadabilityMetricsRaw] = [student_metrics_raw, model_metrics_raw, another_metrics_raw]
    logger.info(f"Normalizing metrics across a corpus of {len(corpus_raw_metrics)} texts...")
    try:
        corpus_normalized_metrics: List[ReadabilityMetricsNormalized] = normalize_metrics(corpus_raw_metrics)
        norm_student_metrics = corpus_normalized_metrics[0]
        norm_model_metrics = corpus_normalized_metrics[1]
        norm_another_metrics = corpus_normalized_metrics[2]
    except ValueError as e_norm:  # Catch errors from normalize_metrics, e.g. StandardScaler issues
        logger.error(f"Failed to normalize metrics: {e_norm}", exc_info=True)
        # Fallback or exit if normalization is critical
        logger.warning("Proceeding with raw metrics for distance calculations due to normalization failure.")
        norm_student_metrics = student_metrics_raw  # type: ignore
        norm_model_metrics = model_metrics_raw  # type: ignore
    except Exception as e_norm_unexpected:
        logger.error(f"Unexpected error during metric normalization: {e_norm_unexpected}", exc_info=True)
        exit(1)

    # --- Compute Distances on Normalized Features ---
    logger.info("Computing distances on normalized features (student vs. model)...")
    euclidean_dist_norm = calculate_euclidean_distance(norm_student_metrics, norm_model_metrics)
    manhattan_dist_norm = calculate_manhattan_distance(norm_student_metrics, norm_model_metrics)

    # --- Output Results ---
    print("\n=== Raw Readability Metrics & Differences (Student vs. Model) ===")
    print(f"{'Metric':<30} | {'Student (Raw)':>15} | {'Model (Raw)':>12} | {'Abs Diff':>10}")
    print("-" * 73)
    # Iterate through fields of ReadabilityMetricsRaw to display
    for key in ReadabilityMetricsRaw.model_fields:
        s_val = getattr(student_metrics_raw, key)
        m_val = getattr(model_metrics_raw, key)
        d_val = getattr(raw_differences, key)  # From MetricDifferencesRaw

        # Format based on whether it's an integer count or float score
        fmt = "{:.0f}" if key in {"difficult_words", "sentence_count", "syllable_count", "lexicon_count"} else "{:.2f}"

        print(
            f"{key.replace('_', ' ').title():<30} | {fmt.format(s_val):>15} | {fmt.format(m_val):>12} | {fmt.format(d_val):>10}"
        )

    print("\n=== Normalized Readability Metrics (Example: Student Text) ===")
    print(f"{'Metric':<30} | {'Normalized Value':>18}")
    print("-" * 53)
    for key in ReadabilityMetricsNormalized.model_fields:
        norm_val = getattr(norm_student_metrics, key)
        print(f"{key.replace('_', ' ').title():<30} | {norm_val:>18.4f}")

    print("\n=== Distances on Normalized Readability Features (Student vs. Model) ===")
    print(f"Euclidean Distance (Normalized): {euclidean_dist_norm:.4f}")
    print(f"Manhattan Distance (Normalized): {manhattan_dist_norm:.4f}")

    # Note: In an essay-scoring pipeline, these distances (raw or normalized)
    # can be valuable features themselves, combined with other text features,
    # or used to assess stylistic similarity/dissimilarity to model essays.
    # Normalization is crucial if these distances are fed into algorithms sensitive to feature scales (e.g., k-NN, SVM).

    logger.info("Readability metrics processing example finished.")
