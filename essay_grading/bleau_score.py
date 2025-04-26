from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# Attempt NLTK imports and provide guidance if missing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:
    msg = "NLTK library not found. Please install it: pip install nltk"
    raise ImportError(msg)

# --- Configuration Defaults ---

# Pre-compile regex for performance
SPECIAL_CHARS_REMOVE_PATTERN = re.compile(r"[^\w\s]")

# Default smoothing method
DEFAULT_SMOOTHING_FUNCTION = SmoothingFunction().method1

# Lazily loaded defaults for NLTK data
_DEFAULT_STOP_WORDS: Optional[Set[str]] = None
_DEFAULT_LEMMATIZER: Optional[WordNetLemmatizer] = None

# --- Helper for NLTK Data ---

_NLTK_DATA_DOWNLOADED = {"punkt": False, "wordnet": False, "stopwords": False}


def _ensure_nltk_data(resource_name: str, download_dir: Optional[str] = None):
    """Check if NLTK resource is available, downloads if not."""
    if _NLTK_DATA_DOWNLOADED.get(resource_name, False):
        return True
    try:
        nltk.data.find(f"tokenizers/{resource_name}" if resource_name == "punkt" else f"corpora/{resource_name}")
        _NLTK_DATA_DOWNLOADED[resource_name] = True
        return True
    except LookupError:
        print(f"NLTK data '{resource_name}' not found. Downloading...")
        try:
            nltk.download(resource_name, download_dir=download_dir, quiet=True)
            _NLTK_DATA_DOWNLOADED[resource_name] = True
            print(f"NLTK data '{resource_name}' downloaded successfully.")
            return True
        except Exception as e:
            warnings.warn(
                f"Failed to download NLTK data '{resource_name}'. "
                f"Preprocessing might fail or be incomplete. Error: {e}",
                RuntimeWarning,
            )
            return False


def _get_default_stop_words() -> Set[str]:
    """Lazily loads default English stopwords."""
    global _DEFAULT_STOP_WORDS
    if _DEFAULT_STOP_WORDS is None:
        if _ensure_nltk_data("stopwords"):
            _DEFAULT_STOP_WORDS = set(stopwords.words("english"))
        else:
            # Fallback to empty set if download fails
            _DEFAULT_STOP_WORDS = set()
    return _DEFAULT_STOP_WORDS


def _get_default_lemmatizer() -> WordNetLemmatizer:
    """Lazily loads default WordNetLemmatizer."""
    global _DEFAULT_LEMMATIZER
    if _DEFAULT_LEMMATIZER is None:
        if _ensure_nltk_data("wordnet"):
            _DEFAULT_LEMMATIZER = WordNetLemmatizer()
        else:
            # Fallback: Create instance anyway, might work partially or raise later
            _DEFAULT_LEMMATIZER = WordNetLemmatizer()
            warnings.warn(
                "WordNet data not found or failed to download. Lemmatization might not work correctly.", RuntimeWarning
            )
    return _DEFAULT_LEMMATIZER


# --- Main Classes ---


@dataclass
class BleuResult:
    """
    Holds BLEU scoring results.

    Attributes:
        score: The overall cumulative BLEU score (typically BLEU-4 with uniform weights unless specified otherwise).
        cumulative_ngram_scores: A dictionary mapping n (from 1 to max_n) to the cumulative BLEU-n score.
                                  For example, key 2 holds the BLEU-2 score (average of 1-gram and 2-gram precision).
                                  This is populated by `score_all_ngrams`.
    """

    score: float
    cumulative_ngram_scores: Optional[Dict[int, float]] = field(default=None)


class BleuScorer:
    """
    Computes BLEU similarity between hypothesis and reference sentences.

    Offers configurable preprocessing (stop words, lemmatization) and
    NLTK smoothing methods. Handles NLTK data downloads automatically.

    Args:
        stop_words: Set of words to exclude during preprocessing.
                      If None, uses default English stopwords from NLTK.
        lemmatizer: A WordNetLemmatizer instance. If None, uses a default
                      NLTK WordNetLemmatizer.
        smoothing_function: NLTK BLEU smoothing method (e.g., SmoothingFunction().method1).
                              If None, uses method1.
        ensure_nltk_data: If True (default), attempts to download required NLTK
                          data ('punkt', 'wordnet', 'stopwords') if not found.
        nltk_download_dir: Optional path to download NLTK data.
    """

    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        lemmatizer: Optional[WordNetLemmatizer] = None,
        smoothing_function: Optional[Callable] = None,
        ensure_nltk_data: bool = True,
        nltk_download_dir: Optional[str] = None,
    ) -> None:
        if ensure_nltk_data:
            _ensure_nltk_data("punkt", download_dir=nltk_download_dir)
            # Lemmatizer and stopwords check/download is handled lazily on first use

        # Assign lemmatizer and stop words, using lazy-loaded defaults if needed
        self.lemmatizer = lemmatizer if lemmatizer is not None else _get_default_lemmatizer()
        self.stop_words = stop_words if stop_words is not None else _get_default_stop_words()

        # Use provided smoothing or default
        self.smoothing = smoothing_function or DEFAULT_SMOOTHING_FUNCTION

        # Basic configuration logging
        logging.info(
            f"BleuScorer initialized. Stop words: {'Default' if stop_words is None else f'{len(stop_words)} custom'}, "
            f"Lemmatizer: {'Default' if lemmatizer is None else 'Custom'}, "
            f"Smoothing: {self.smoothing.__name__ if hasattr(self.smoothing, '__name__') else 'Custom Function'}"
        )

    @lru_cache(maxsize=512)  # Increased cache size slightly
    def _preprocess_text(self, text: str) -> Tuple[str, ...]:
        """
        Tokenizes, cleans, lemmatizes, and filters stop words from text.

        Returns a tuple of processed tokens, suitable for caching.

        Args:
            text: The input sentence string.

        Returns:
            A tuple of processed token strings.
        """
        try:
            # 1. Remove special characters and lowercase
            cleaned = SPECIAL_CHARS_REMOVE_PATTERN.sub("", text.lower())
            # 2. Tokenize
            tokens = word_tokenize(cleaned)
            # 3. Lemmatize and filter stop words
            processed_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalnum() and token not in self.stop_words  # Ensure alphanumeric and not stop word
            ]
            return tuple(processed_tokens)
        except Exception as e:
            logging.error(f"Error during preprocessing text: '{text[:50]}...'. Error: {e}")
            # Return empty tuple on failure to allow BLEU to compute (likely 0)
            return tuple()

    def score(
        self,
        references: Union[str, Sequence[str]],
        hypothesis: str,
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),  # Default BLEU-4
    ) -> BleuResult:
        """
        Computes the cumulative BLEU score with specified n-gram weights.

        Args:
            references: The reference sentence(s). Can be a single string or
                        a list/tuple of strings.
            hypothesis: The hypothesis sentence to score.
            weights: A tuple of weights for n-grams (e.g., (0.25, 0.25, 0.25, 0.25)
                     for standard BLEU-4). The length determines the maximum n-gram order.

        Returns:
            A BleuResult object containing the overall score.
        """
        if isinstance(references, str):
            reference_list = [references]
        elif isinstance(references, (list, tuple)):
            reference_list = references
        else:
            raise TypeError("References must be a string or a sequence of strings.")

        if not hypothesis or not reference_list or not any(reference_list):
            logging.warning("Cannot compute BLEU score with empty reference(s) or hypothesis.")
            return BleuResult(score=0.0)

        try:
            # Preprocess hypothesis
            hyp_tokens: List[str] = list(self._preprocess_text(hypothesis))

            # Preprocess reference(s)
            ref_tokens_list: List[List[str]] = [list(self._preprocess_text(ref)) for ref in reference_list]

            # Handle cases where preprocessing results in empty lists
            if not hyp_tokens or not any(ref_tokens_list):
                score_value = 0.0
                if not hyp_tokens:
                    logging.warning(f"Hypothesis '{hypothesis[:50]}...' became empty after preprocessing.")
                if not any(ref_tokens_list):
                    logging.warning(f"All references became empty after preprocessing.")

            else:
                score_value = sentence_bleu(
                    ref_tokens_list, hyp_tokens, weights=weights, smoothing_function=self.smoothing
                )

            return BleuResult(score=score_value)

        except Exception as e:
            # Log specific inputs that caused the error for easier debugging
            ref_repr = (
                f"'{reference_list[0][:50]}...'"
                if isinstance(reference_list, list) and reference_list
                else str(reference_list)
            )
            hyp_repr = f"'{hypothesis[:50]}...'"
            logging.exception(
                f"Error computing BLEU score for refs: {ref_repr} and hyp: {hyp_repr}. Weights: {weights}",
                exc_info=e,  # Log the full traceback
            )
            return BleuResult(score=0.0)  # Return 0 score on failure

    def score_all_ngrams(self, references: Union[str, Sequence[str]], hypothesis: str, max_n: int = 4) -> BleuResult:
        """
        Computes cumulative BLEU scores for each n-gram order up to max_n,
        plus an overall score using uniform weights up to max_n.

        Args:
            references: The reference sentence(s). Can be a single string or
                        a list/tuple of strings.
            hypothesis: The hypothesis sentence.
            max_n: The maximum n-gram size to evaluate (default 4).

        Returns:
            A BleuResult containing the overall score (uniform weights up to max_n)
            and a dictionary `cumulative_ngram_scores` mapping n (1 to max_n)
            to the cumulative BLEU-n score.
        """
        if isinstance(references, str):
            reference_list = [references]
        elif isinstance(references, (list, tuple)):
            reference_list = references
        else:
            raise TypeError("References must be a string or a sequence of strings.")

        if not hypothesis or not reference_list or not any(reference_list):
            logging.warning("Cannot compute BLEU score with empty reference(s) or hypothesis.")
            return BleuResult(score=0.0, cumulative_ngram_scores={n: 0.0 for n in range(1, max_n + 1)})

        try:
            # Preprocess hypothesis
            hyp_tokens: List[str] = list(self._preprocess_text(hypothesis))

            # Preprocess reference(s)
            ref_tokens_list: List[List[str]] = [list(self._preprocess_text(ref)) for ref in reference_list]

            # Handle cases where preprocessing results in empty lists
            if not hyp_tokens or not any(ref_tokens_list):
                if not hyp_tokens:
                    logging.warning(f"Hypothesis '{hypothesis[:50]}...' became empty after preprocessing.")
                if not any(ref_tokens_list):
                    logging.warning(f"All references became empty after preprocessing.")
                return BleuResult(score=0.0, cumulative_ngram_scores={n: 0.0 for n in range(1, max_n + 1)})

            cumulative_scores: Dict[int, float] = {}
            for n in range(1, max_n + 1):
                # Weights for cumulative BLEU-n: uniform up to n, zero beyond
                weights = tuple(1.0 / n if i < n else 0.0 for i in range(max_n))
                ngram_score = sentence_bleu(
                    ref_tokens_list, hyp_tokens, weights=weights, smoothing_function=self.smoothing
                )
                cumulative_scores[n] = ngram_score  # No need for float() cast

            # Calculate overall score using uniform weights across all considered n-grams
            uniform_weights = tuple(1.0 / max_n for _ in range(max_n))
            overall_score = sentence_bleu(
                ref_tokens_list, hyp_tokens, weights=uniform_weights, smoothing_function=self.smoothing
            )

            return BleuResult(score=overall_score, cumulative_ngram_scores=cumulative_scores)

        except Exception as e:
            ref_repr = (
                f"'{reference_list[0][:50]}...'"
                if isinstance(reference_list, list) and reference_list
                else str(reference_list)
            )
            hyp_repr = f"'{hypothesis[:50]}...'"
            logging.exception(
                f"Error computing n-gram BLEU scores for refs: {ref_repr} and hyp: {hyp_repr}. Max_n: {max_n}",
                exc_info=e,
            )
            # Return 0 scores on failure
            return BleuResult(score=0.0, cumulative_ngram_scores={n: 0.0 for n in range(1, max_n + 1)})


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Basic Usage ---
    scorer = BleuScorer()  # Uses defaults, will trigger NLTK downloads if needed

    ref1 = "The quick brown fox jumps over the lazy dog"
    hyp1 = "the fast brown fox jumped over the sleepy dog"
    hyp2 = "this is a completely different sentence"

    result1 = scorer.score(ref1, hyp1)
    print(f"\nScore (Ref1 vs Hyp1): {result1.score:.4f}")

    result2 = scorer.score(ref1, hyp2)
    print(f"Score (Ref1 vs Hyp2): {result2.score:.4f}")

    # --- Multiple References ---
    refs = ["The quick brown fox jumps over the lazy dog", "A fast brown fox leaps over the sleepy canine"]
    result_multi = scorer.score(refs, hyp1)
    print(f"Score (Multi-Ref vs Hyp1): {result_multi.score:.4f}")

    # --- N-gram Breakdown ---
    result_ngrams = scorer.score_all_ngrams(ref1, hyp1, max_n=4)
    print(f"\nN-gram breakdown (Ref1 vs Hyp1):")
    print(f"  Overall Score (Uniform Weights 1-4): {result_ngrams.score:.4f}")
    if result_ngrams.cumulative_ngram_scores:
        for n, score in result_ngrams.cumulative_ngram_scores.items():
            print(f"  Cumulative BLEU-{n}: {score:.4f}")

    # --- Custom Configuration ---
    custom_stopwords = {"the", "a", "is", "over"}
    custom_smoothing = SmoothingFunction().method7  # Different smoothing
    custom_scorer = BleuScorer(
        stop_words=custom_stopwords,
        smoothing_function=custom_smoothing,
        ensure_nltk_data=False,  # Assume data is present if set to False
    )
    result_custom = custom_scorer.score(ref1, hyp1)
    print(f"\nScore with custom settings (Ref1 vs Hyp1): {result_custom.score:.4f}")

    # --- Edge Case: Empty Input ---
    print("\nTesting edge case (empty hypothesis):")
    result_empty = scorer.score(ref1, "")
    print(f"Score (Ref1 vs Empty Hyp): {result_empty.score:.4f}")  # Should be 0.0

    print("\nTesting edge case (empty reference):")
    result_empty_ref = scorer.score("", hyp1)
    print(f"Score (Empty Ref vs Hyp1): {result_empty_ref.score:.4f}")  # Should be 0.0

    print("\nTesting edge case (text becomes empty after preprocessing):")
    ref_stops = "a the is"
    hyp_stops = "the a is"
    result_empty_processed = scorer.score(ref_stops, hyp_stops)
    print(f"Score (Stopwords only): {result_empty_processed.score:.4f}")  # Should be 0.0
