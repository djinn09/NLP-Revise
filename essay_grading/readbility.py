"""Module provides utilities for calculating and analyzing readability metrics.

It includes:
- Pydantic models for raw and normalized readability metrics.
- Functions to compute, normalize, and compare readability metrics.
- Distance calculation methods for comparing readability features.
"""

# Use annotations for cleaner type hinting (requires Python 3.7+)
from __future__ import annotations

import logging
import math  # Re-added for isnan/isinf checks
import sys  # Added for sys.exit()
from typing import Union  # Dict, List, Optional removed as unused or replaced by builtins

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
    """Base model for readability metrics.

    Defines common fields and validators. This can be inherited by Raw and
    Normalized metric models.
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

    @field_validator(
        "difficult_words",
        "sentence_count",
        "syllable_count",
        "lexicon_count",
        mode="before",  # Validate before Pydantic's type conversion
    )
    @classmethod
    def ensure_integer_counts(cls, value: float) -> int:  # Changed Union[int, float] to float
        """Ensure count metrics are integers. Textstat might return floats."""
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                msg = f"Count metric cannot be NaN or infinity, got {value}"
                raise ValueError(msg)  # Keep ValueError for NaN/inf as it's about value, not type
            return round(value)
        if isinstance(value, int):
            return value  # Corrected: no redundant int() cast
        msg = f"Count metric must be a number (int or float), got {type(value)}"
        raise TypeError(msg)  # Changed to TypeError

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
    def ensure_float_scores(cls, value: float) -> float:  # Changed Union[int, float] to float
        """Ensure score metrics are floats."""
        if not isinstance(value, (int, float)):  # Keep check for int and float
            msg = f"Score metric must be a number, got {type(value)}"
            raise TypeError(msg)  # Changed to TypeError
        return float(value)


class ReadabilityMetricsRaw(BaseReadabilityMetrics):
    """Data model for storing raw readability metrics.

    Calculated for a given text. Inherits fields and validators from
    BaseReadabilityMetrics.
    """


class ReadabilityMetricsNormalized(BaseReadabilityMetrics):
    """Data model for storing normalized readability metrics.

    E.g., standardized. The values will be scaled, but the structure
    matches the raw metrics. Inherits fields and validators from
    BaseReadabilityMetrics.
    Note: ge=0 validation for counts might not apply strictly after
    normalization if normalization can result in negative values
    (e.g. standard scaler). We can override or remove those specific
    field constraints for normalized data if needed. For simplicity
    here, we keep them, assuming normalization mostly scales around 0.
    """

    difficult_words: float = Field(..., description="Normalized count of difficult words.")
    sentence_count: float = Field(..., description="Normalized count of sentences.")
    avg_sentence_length: float = Field(..., description="Normalized average words per sentence.")
    syllable_count: float = Field(..., description="Normalized total number of syllables.")
    lexicon_count: float = Field(..., description="Normalized total words (punctuation removed).")


class MetricDifferencesRaw(BaseModel):
    """Data model for storing absolute differences.

    Between two sets of raw readability metrics. All difference values
    are non-negative floats.
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
    """Compute a set of raw readability metrics for a given text.

    Uses the `textstat` library.

    Args:
        text: The input string. An empty string or string with insufficient
              content might lead to errors or default/zero values from
              `textstat` functions.

    Returns:
        A ReadabilityMetricsRaw Pydantic model instance containing the
        calculated scores.

    Raises:
        TypeError: If the input `text` is not a string.
        Exception: Propagates exceptions from `textstat` if critical errors
                   occur (e.g., due to unsupported language if
                   `textstat.set_lang` was used incorrectly, or internal
                   errors for highly unusual inputs).

    """
    if not isinstance(text, str):
        logger.error("Input 'text' for readability metrics must be a string.")
        # EM101: assign to variable first; TRY003: avoid long messages
        msg = "Input 'text' must be a string."
        raise TypeError(msg)

    if not text.strip():
        logger.warning("Input text is empty or whitespace only. Readability metrics will likely be defaults or zeros.")
        return ReadabilityMetricsRaw(
            flesch_reading_ease=0.0,
            flesch_kincaid_grade=0.0,
            smog_index=0.0,
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
        raw_metrics_dict = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),  # type: ignore[attr-defined]
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),  # type: ignore[attr-defined]
            "smog_index": textstat.smog_index(text),  # type: ignore[attr-defined]
            "gunning_fog": textstat.gunning_fog(text),  # type: ignore[attr-defined]
            "dale_chall": textstat.dale_chall_readability_score(text),  # type: ignore[attr-defined]
            "automated_readability_index": textstat.automated_readability_index(text),  # type: ignore[attr-defined]
            "coleman_liau_index": textstat.coleman_liau_index(text),  # type: ignore[attr-defined]
            "linsear_write_formula": textstat.linsear_write_formula(text),  # type: ignore[attr-defined]
            "difficult_words": textstat.difficult_words(text),  # type: ignore[attr-defined]
            "sentence_count": textstat.sentence_count(text),  # type: ignore[attr-defined]
            "avg_sentence_length": textstat.avg_sentence_length(text),  # type: ignore[attr-defined]
            "syllable_count": textstat.syllable_count(text),  # type: ignore[attr-defined]
            "lexicon_count": textstat.lexicon_count(text, removepunct=True),  # type: ignore[attr-defined]
        }
        return ReadabilityMetricsRaw(**raw_metrics_dict)
    except Exception:
        # G201: Use logger.exception
        logger.exception(f"Error calculating readability metrics for text: '{text[:50]}...'")
        raise


def normalize_metrics(metrics_list: list[ReadabilityMetricsRaw]) -> list[ReadabilityMetricsNormalized]:
    """Standardize each readability feature across a list of texts.

    Uses `sklearn.preprocessing.StandardScaler` to achieve zero mean
    and unit variance.

    Args:
        metrics_list: A list of ReadabilityMetricsRaw Pydantic model
                      instances, where each instance represents the raw
                      metrics for one text.

    Returns:
        A list of ReadabilityMetricsNormalized Pydantic model instances,
        with metrics scaled. Returns an empty list if the input is empty.
        If input list contains only one item, StandardScaler will result
        in all zeros.

    """
    if not metrics_list:
        logger.warning("normalize_metrics received an empty list. Returning empty list.")
        return []
    if not all(isinstance(m, ReadabilityMetricsRaw) for m in metrics_list):
        msg = "All items in metrics_list must be ReadabilityMetricsRaw instances."
        raise TypeError(msg)

    keys = list(ReadabilityMetricsRaw.model_fields.keys())

    try:
        mat = np.array(
            [[getattr(m, key) for key in keys] for m in metrics_list],
            dtype=float,  # COM812: Trailing comma
        )
    except Exception as e:
        logger.exception("Error converting metrics list to NumPy array")
        msg = "Could not convert metrics list to NumPy array. Check metric values."
        raise ValueError(msg) from e

    if mat.shape[0] < 1:
        return []

    scaler = StandardScaler()
    if mat.shape[0] == 1:
        logger.warning(
            "Normalizing metrics with only one sample. "
            "StandardScaler will produce all zeros for the normalized metrics.",
        )
        mat_norm = np.zeros_like(mat, dtype=float)
    else:
        try:
            mat_norm = scaler.fit_transform(mat)
        except ValueError as e_scale:
            log_msg = (  # E501: Line too long
                f"Error during StandardScaler fit_transform: {e_scale}. "
                "This can happen if a feature has zero variance "
                "(all values are the same)."
            )
            logger.exception(log_msg)
            # EM102 (f-string), TRY003 (long message)
            err_msg = f"StandardScaler failed, possibly due to zero variance in a feature: {e_scale}"
            raise ValueError(err_msg) from e_scale

    normalized_metrics_obj_list: list[ReadabilityMetricsNormalized] = []
    for row in mat_norm:
        # B905: Add strict=True to zip
        norm_dict = dict(zip(keys, row, strict=True))
        try:
            normalized_metrics_obj_list.append(ReadabilityMetricsNormalized(**norm_dict))
        except Exception as e_pydantic:
            log_msg = (  # E501: Line too long
                f"Error creating ReadabilityMetricsNormalized model from "
                f"normalized data: {norm_dict}. Error: {e_pydantic}"
            )
            logger.exception(log_msg)
            msg = "Failed to reconstruct Pydantic model for normalized metrics."
            raise ValueError(msg) from e_pydantic

    return normalized_metrics_obj_list


def calculate_euclidean_distance(
    metrics1: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
    metrics2: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
) -> float:
    """Compute the Euclidean (L2) distance between two feature vectors.

    Feature vectors are represented by ReadabilityMetrics Pydantic models.

    Args:
        metrics1: Readability metrics for the first text.
        metrics2: Readability metrics for the second text.

    Returns:
        The Euclidean distance as a float.

    """
    if not isinstance(metrics1, BaseReadabilityMetrics) or not isinstance(metrics2, BaseReadabilityMetrics):
        msg = "Inputs must be instances of a BaseReadabilityMetrics model."
        raise TypeError(msg)

    keys = list(BaseReadabilityMetrics.model_fields.keys())
    arr1 = np.array([getattr(metrics1, key) for key in keys], dtype=float)
    arr2 = np.array([getattr(metrics2, key) for key in keys], dtype=float)

    distance = np_norm(arr1 - arr2)
    return float(distance)


def calculate_manhattan_distance(
    metrics1: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],
    metrics2: Union[ReadabilityMetricsRaw, ReadabilityMetricsNormalized],  # COM812: Trailing comma
) -> float:
    """Compute the Manhattan (L1) distance between two feature vectors.

    Feature vectors are represented by ReadabilityMetrics Pydantic models.

    Args:
        metrics1: Readability metrics for the first text.
        metrics2: Readability metrics for the second text.

    Returns:
        The Manhattan distance as a float.

    """
    if not isinstance(metrics1, BaseReadabilityMetrics) or not isinstance(metrics2, BaseReadabilityMetrics):
        msg = "Inputs must be instances of a BaseReadabilityMetrics model."
        raise TypeError(msg)

    keys = list(BaseReadabilityMetrics.model_fields.keys())
    arr1 = np.array([getattr(metrics1, key) for key in keys], dtype=float)
    arr2 = np.array([getattr(metrics2, key) for key in keys], dtype=float)

    distance = np.abs(arr1 - arr2).sum()  # COM812 for this line is likely a linter error, not fixed.
    return float(distance)


def compare_raw_metrics_absolute_diff(
    metrics1: ReadabilityMetricsRaw,
    metrics2: ReadabilityMetricsRaw,  # COM812: Trailing comma
) -> MetricDifferencesRaw:
    """Compute absolute differences between raw scores.

    Scores are from two ReadabilityMetricsRaw objects. This is useful
    for direct interpretability of score differences.

    Args:
        metrics1: Raw readability metrics for the first text.
        metrics2: Raw readability metrics for the second text.

    Returns:
        A MetricDifferencesRaw Pydantic model instance containing the
        absolute differences.

    """
    if not isinstance(metrics1, ReadabilityMetricsRaw) or not isinstance(metrics2, ReadabilityMetricsRaw):
        msg = "Inputs must be ReadabilityMetricsRaw instances for raw comparison."
        raise TypeError(msg)

    m1_dict = metrics1.model_dump()
    m2_dict = metrics2.model_dump()

    differences_dict = {key: abs(m1_dict[key] - m2_dict[key]) for key in m1_dict}
    return MetricDifferencesRaw(**differences_dict)


# --- Main Execution Block ---
if __name__ == "__main__":
    logger.info("Starting readability metrics processing example.")

    # E501: Lines too long
    student_text = (
        "Education is the passport to the future, for tomorrow belongs to "
        "those who prepare for it today. The journey of learning is lifelong."
    )
    model_text = (
        "The future belongs to those who prepare for it today; "
        "education is their passport. Learning is a continuous voyage."
    )
    another_text = "This is a very simple and short text. It has few words. Reading is easy."

    try:
        logger.info("Computing raw metrics for student text...")
        student_metrics_raw = get_readability_metrics(student_text)
        logger.info("Computing raw metrics for model text...")
        model_metrics_raw = get_readability_metrics(model_text)
        logger.info("Computing raw metrics for another text...")
        another_metrics_raw = get_readability_metrics(another_text)
    except Exception:
        logger.exception("Failed to compute raw readability metrics:")
        sys.exit(1)  # PLR1722: Use sys.exit()

    logger.info("Comparing raw metrics (student vs. model)...")
    raw_differences = compare_raw_metrics_absolute_diff(student_metrics_raw, model_metrics_raw)

    corpus_raw_metrics: list[ReadabilityMetricsRaw] = [student_metrics_raw, model_metrics_raw, another_metrics_raw]
    logger.info(f"Normalizing metrics across a corpus of {len(corpus_raw_metrics)} texts...")
    try:
        corpus_normalized_metrics: list[ReadabilityMetricsNormalized] = normalize_metrics(corpus_raw_metrics)
        len_of_normalized_metrics = 3
        # Ensure we have enough items before indexing
        if len(corpus_normalized_metrics) < len_of_normalized_metrics:
            logger.error("Normalization resulted in fewer metrics than expected. Exiting.")
            sys.exit(1)
        norm_student_metrics = corpus_normalized_metrics[0]
        norm_model_metrics = corpus_normalized_metrics[1]
        # norm_another_metrics = corpus_normalized_metrics[2] # Not used later, but good to have
    except ValueError:
        logger.exception("Failed to normalize metrics:")
        logger.warning("Proceeding with raw metrics for distance calculations due to normalization failure.")
        # Ensure types are compatible if falling back. This is tricky.
        # For this example, we'll assume they are, but in prod this needs care.
        norm_student_metrics = student_metrics_raw  # type: ignore[assignment]
        norm_model_metrics = model_metrics_raw  # type: ignore[assignment]
    except Exception:
        logger.exception("Unexpected error during metric normalization:")
        sys.exit(1)  # PLR1722: Use sys.exit()

    logger.info("Computing distances on normalized features (student vs. model)...")
    euclidean_dist_norm = calculate_euclidean_distance(norm_student_metrics, norm_model_metrics)
    manhattan_dist_norm = calculate_manhattan_distance(norm_student_metrics, norm_model_metrics)

    print("\n=== Raw Readability Metrics & Differences (Student vs. Model) ===")
    print(f"{'Metric':<30} | {'Student (Raw)':>15} | {'Model (Raw)':>12} | {'Abs Diff':>10}")
    print("-" * 73)
    for key in ReadabilityMetricsRaw.model_fields:
        s_val = getattr(student_metrics_raw, key)
        m_val = getattr(model_metrics_raw, key)
        d_val = getattr(raw_differences, key)

        fmt = "{:.0f}" if key in {"difficult_words", "sentence_count", "syllable_count", "lexicon_count"} else "{:.2f}"
        # E501: Line too long, COM812 (related to print call if multi-line)
        line_to_print = (
            f"{key.replace('_', ' ').title():<30} | "
            f"{fmt.format(s_val):>15} | "
            f"{fmt.format(m_val):>12} | "
            f"{fmt.format(d_val):>10}"
        )
        print(line_to_print)

    print("\n=== Normalized Readability Metrics (Example: Student Text) ===")
    print(f"{'Metric':<30} | {'Normalized Value':>18}")
    print("-" * 53)
    for key in ReadabilityMetricsNormalized.model_fields:
        # Check if norm_student_metrics is BaseReadabilityMetrics, which might not have all normalized fields as float
        # This can happen if normalization failed and it fell back to raw metrics.
        # For this example, we assume norm_student_metrics is ReadabilityMetricsNormalized or compatible.
        norm_val = getattr(norm_student_metrics, key)
        print(f"{key.replace('_', ' ').title():<30} | {norm_val:>18.4f}")

    print("\n=== Distances on Normalized Readability Features (Student vs. Model) ===")
    print(f"Euclidean Distance (Normalized): {euclidean_dist_norm:.4f}")
    print(f"Manhattan Distance (Normalized): {manhattan_dist_norm:.4f}")

    logger.info("Readability metrics processing example finished.")
