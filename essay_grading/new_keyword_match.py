from __future__ import annotations

import difflib
import logging
import os
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import nltk
import rapidfuzz
from fuzzywuzzy import fuzz as fuzzy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rank_bm25 import BM25L as BM25
from rapidfuzz import fuzz
from scipy.sparse import csr_matrix
from similarity.cosine import Cosine
from similarity.jaccard import Jaccard
from similarity.jarowinkler import JaroWinkler
from similarity.metric_lcs import MetricLCS
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.qgram import QGram
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_score
from sklearn.metrics.pairwise import (
    cosine_similarity,
    pairwise_distances,
)

# try:
#     nltk.data.find("tokenizers/punkt")
#     nltk.data.find("corpora/wordnet.zip/wordnet/")
#     nltk.data.find("corpora/stopwords")
#     nltk.data.find("corpora/omw.zip")
#     nltk.data.find("tokenizers/punkt_tab")
# except Exception:
#     nltk.download("punkt")
#     nltk.download("wordnet")  # Download WordNet for lemmatization
#     nltk.download('omw')
#     nltk.download("stopwords")
#     nltk.download("punkt_tab")


stemmer = PorterStemmer()
word_net_lemmatizer = WordNetLemmatizer()  # Create WordNet lemmatizer
remove_punctuation_map = {ord(char): None for char in string.punctuation}

STOPWORDS = set(stopwords.words("english"))


# Default smoothing method
DEFAULT_SMOOTHING_FUNCTION = SmoothingFunction().method1
normalized_levenshtein = NormalizedLevenshtein()
jaro_winkler = JaroWinkler()
metric_lcs = MetricLCS()
qgram2 = QGram(2)  # QGram with n=2
qgram3 = QGram(3)  # QGram with n=3
qgram4 = QGram(4)  # QGram with n=4
sim_cosine = Cosine(2)
jaccard = Jaccard(2)

SPECIAL_CHARS_REMOVE_PATTERN = r'[-()"#/@&^*();:<>{}`+=~|!?,]'  # Pattern to remove special characters


class TFIDF:
    """Compute TF-IDF based similarity or distance metrics between two texts.

    Attributes
    ----------
    original : str
        The original text for comparison.
    compare_text : str
        The text to compare against the original.
    without_normalized : bool
        If True, skip custom normalization and use regex token pattern.
    lemmatizer : Callable
        The NLTK lemmatizer used for lemmatization.
    normalize : Optional[Callable]
        The normalization function (stem or lemma) or None if skipped.
    vectorizer_ : TfidfVectorizer
        The configured TF-IDF vectorizer.

    Methods
    -------
    lemmatize_tokens(tokens)
        Lemmatizes a list of tokens.
    stem_normalize(text)
        Normalizes text by stemming and removing punctuation.
    lemma_normalize(text)
        Normalizes text by lemmatizing and removing punctuation.
    vectorizer(without_normalized)
        Returns a TF-IDF vectorized based on the normalization setting.
    fit()
        Fits the TF-IDF vectorized on the original and comparison texts.
    calculate_distance(metric)
        Calculates the similarity or distance between the texts using the specified metric.

    """

    def __init__(
        self,
        original: str,
        compare_text: str,
        *,
        lemmatization: bool = False,
        without_normalized: bool = False,
        lemmatizer: Optional[Callable] = None,
    ) -> None:
        """Initialize the TFIDF instance.

        Parameters
        ----------
        original : str
            The original text.
        compare_text : str
            The text to compare.
        lemmatization : bool, optional
            If True, use lemmatization for normalization (default False).
        without_normalized : bool, optional
            If True, skip custom normalization (default False).
        lemmatizer : Callable, optional
            Custom lemmatizer to use (default uses WordNetLemmatizer).

        """
        self.original = original
        self.compare_text = compare_text
        self.without_normalized = without_normalized

        # Assign the lemmatizer instance (default if none provided)
        self.lemmatizer = lemmatizer or word_net_lemmatizer

        # Decide on normalization: stem, lemma, or None (if skipped)
        if not without_normalized:
            self.normalize = self.lemma_normalize if lemmatization else self.stem_normalize
        else:
            self.normalize = None

        # Create and cache the TF-IDF vectorizer
        self.vectorizer_ = self._create_vectorizer()

    def _basic_tokenize(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, and tokenize the text.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        List[str]
            List of word tokens.

        """
        # Remove punctuation and tokenize
        return nltk.word_tokenize(text.lower().translate(remove_punctuation_map))

    def stem_normalize(self, text: str) -> List[str]:
        """Normalize text by stemming tokens.

        Parameters
        ----------
        text : str
            Input text to normalize.

        Returns
        -------
        List[str]
            List of stemmed tokens.

        """
        tokens = self._basic_tokenize(text)
        # Apply Porter stemmer to each token
        return [stemmer.stem(token) for token in tokens]

    def lemma_normalize(self, text: str) -> List[str]:
        """Normalize text by lemmatizing tokens.

        Parameters
        ----------
        text : str
            Input text to normalize.

        Returns
        -------
        List[str]
            List of lemmatized tokens.

        """
        tokens = self._basic_tokenize(text)
        # Use the provided lemmatizer instance
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def _create_vectorizer(self) -> TfidfVectorizer:
        """Create the TF-IDF vectorizer based on normalization settings.

        Returns
        -------
        TfidfVectorizer
            Configured vectorizer (with custom tokenizer or token pattern).

        """
        if self.without_normalized:
            # Use default regex pattern for tokenization
            token_pattern = os.getenv("TOKEN_PATTERN", r"\w+")
            return TfidfVectorizer(token_pattern=token_pattern)
        # Use custom normalize function
        return TfidfVectorizer(tokenizer=self.normalize, token_pattern=None)  # type: ignore  # noqa: PGH003

    def fit(self) -> csr_matrix:
        """Fit the TF-IDF vectorizer on the original and compare texts.

        Returns
        -------
        scipy.sparse.csr_matrix
            TF-IDF matrix for both texts.

        """
        result = self.vectorizer_.fit_transform([self.original, self.compare_text])
        if not isinstance(result, csr_matrix):
            msg = "Expected csr_matrix but got {type(result).__name__}"
            raise TypeError(msg)
        return result

    def get_normalized_tokens(self) -> Tuple[List[str], List[str]]:
        """Get normalized token lists for debugging or testing.

        Returns
        -------
        Tuple[List[str], List[str]]
            Normalized tokens of original and compare_text.

        Raises
        ------
        ValueError
            If normalization is disabled.

        """
        if not self.normalize:
            msg = "Normalization is disabled."
            raise ValueError(msg)
        return (self.normalize(self.original), self.normalize(self.compare_text))

    def calculate_distance(self, metric: str) -> float:
        """Compute the specified similarity/distance metric between texts.

        Parameters
        ----------
        metric : str
            One of 'cosine', 'euclidean', 'manhattan', 'minkowski', 'jaccard', 'hamming'.

        Returns
        -------
        float
            The computed similarity or distance.

        Raises
        ------
        ValueError
            If an unsupported metric is provided.

        """
        tfidf = self.fit()
        m = metric.lower()
        if m not in ("cosine", "euclidean", "manhattan", "minkowski", "jaccard", "hamming"):
            msg = f"Unsupported metric: {metric}"
            raise ValueError(msg)
        if m == "cosine":
            print("TFIDF Cosine similarity")  # noqa: T201
            # Cosine similarity in [0,1]
            # Compute full pairwise cosine similarity matrix (2x2)
            sim_matrix = cosine_similarity(tfidf)
            # Return similarity between text0 and text1
            return float(sim_matrix[0, 1])

        if m in {"euclidean", "manhattan", "minkowski"}:
            # Pairwise distances from TF-IDF vectors
            return float(pairwise_distances(tfidf.toarray(), metric=m)[0, 1])

        if m == "jaccard":
            # Summarizing my current belief:
            # - Micro-Jaccard: good if you want a "how much vocab overlaps" score.
            # - Weighted-Jaccard: better for longer texts where important words matter more.
            # - Macro-Jaccard: risky and too noisy.
            # - Binary-Jaccard: acceptable if TF-IDF already filtered out unimportant words.
            # Compute Jaccard distance using sklearn's jaccard_score
            print("TFIDF Jaccard distance...")  # noqa: T201
            # Convert TF-IDF sparse matrix to binary presence matrix
            dense = tfidf.toarray()
            presence = (dense > 0).astype(int)
            # jaccard_score returns similarity; subtract from 1 to get distance
            # score = jaccard_score(presence[0], presence[1], average="micro")
            # print(f"micro-Jaccard score: {score}")
            # score = jaccard_score(presence[0], presence[1], average="macro")
            # print(f"macro-Jaccard score: {score}")
            score = jaccard_score(presence[0], presence[1], average="weighted")
            # score = jaccard_score(presence[0], presence[1], average="binary")
            # print(f"binary-Jaccard score: {score}")

            return 1.0 - float(score)

        if m == "hamming":
            print("Hamming distance")  # noqa: T201
            # Character histogram distance normalized by max length
            c1, c2 = Counter(self.original), Counter(self.compare_text)
            all_chars = set(c1) | set(c2)
            diff_sum = sum(abs(c1[ch] - c2[ch]) for ch in all_chars)
            max_len = max(len(self.original), len(self.compare_text), 1)
            return diff_sum / max_len

        return 0.0  # Default case, should not reach here

    def __repr__(self) -> str:
        """Representation showing lengths and normalization state.

        Returns
        -------
        str
            Informative string representation.

        """
        return (
            f"<TFIDF(len_orig={len(self.original)}, "
            f"len_cmp={len(self.compare_text)}, "
            f"normalized={self.normalize is not None})>"
        )


@dataclass
class BleuResult:
    """Holds BLEU scoring results.

    Attributes:
        score: The overall cumulative BLEU score (typically BLEU-4 with uniform weights unless specified otherwise).
        cumulative_ngram_scores: A dictionary mapping n (from 1 to max_n) to the cumulative BLEU-n score.
                                  For example, key 2 holds the BLEU-2 score (average of 1-gram and 2-gram precision).
                                  This is populated by `score_all_ngrams`.

    """

    score: float
    cumulative_ngram_scores: Optional[Dict[int, float]] = field(default=None)


class BleuScorer:
    """Computes BLEU similarity between hypothesis and reference sentences.

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
    ) -> None:
        # Lemmatizer and stopwords check/download is handled lazily on first use

        # Assign lemmatizer and stop words, using lazy-loaded defaults if needed
        self.lemmatizer = lemmatizer or word_net_lemmatizer
        self.stop_words = stop_words or STOPWORDS

        # Use provided smoothing or default
        self.smoothing = smoothing_function or DEFAULT_SMOOTHING_FUNCTION

        # Basic configuration logging
        logging.info(
            f"BleuScorer initialized. Stop words: {'Default' if stop_words is None else f'{len(stop_words)} custom'}, "  # noqa: G004
            f"Lemmatizer: {'Default' if lemmatizer is None else 'Custom'}, "
            f"Smoothing: {self.smoothing.__name__ if hasattr(self.smoothing, '__name__') else 'Custom Function'}",
        )

    @lru_cache(maxsize=512)  # Increased cache size slightly  # noqa: B019
    def _preprocess_text(self, text: str) -> Tuple[str, ...]:
        """Tokenizes, cleans, lemmatizes, and filters stop words from text.

        Returns a tuple of processed tokens, suitable for caching.

        Args:
            text: The input sentence string.

        Returns:
            A tuple of processed token strings.

        """
        try:
            # 1. Remove special characters and lowercase
            cleaned = re.sub(SPECIAL_CHARS_REMOVE_PATTERN, "", text.lower())
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
            logging.exception(f"Error during preprocessing text: '{text[:50]}...'. Error: {e}")  # noqa: G004, LOG015, TRY401
            # Return empty tuple on failure to allow BLEU to compute (likely 0)
            return ()

    def score(
        self,
        references: Union[str, Sequence[str]],
        hypothesis: str,
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),  # Default BLEU-4
    ) -> BleuResult:
        """Compute the cumulative BLEU score with specified n-gram weights.

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
            msg = "References must be a string or a sequence of strings."
            raise TypeError(msg)

        if not hypothesis or not reference_list or not any(reference_list):
            logging.warning("Cannot compute BLEU score with empty reference(s) or hypothesis.")  # noqa: LOG015
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
                    logging.warning(f"Hypothesis '{hypothesis[:50]}...' became empty after preprocessing.")  # noqa: G004, LOG015
                if not any(ref_tokens_list):
                    logging.warning("All references became empty after preprocessing.")  # noqa: LOG015

            else:
                score_value = sentence_bleu(
                    ref_tokens_list,
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing,
                )

            return BleuResult(score=score_value)  # type: ignore  # noqa: PGH003

        except Exception as e:
            # Log specific inputs that caused the error for easier debugging
            ref_repr = (
                f"'{reference_list[0][:50]}...'"
                if isinstance(reference_list, list) and reference_list
                else str(reference_list)
            )
            hyp_repr = f"'{hypothesis[:50]}...'"
            logging.exception(  # noqa: LOG015
                f"Error computing BLEU score for refs: {ref_repr} and hyp: {hyp_repr}. Weights: {weights}",  # noqa: G004
                exc_info=e,  # Log the full traceback
            )
            return BleuResult(score=0.0)  # Return 0 score on failure

    def score_all_ngrams(
        self,
        references: Union[str, Sequence[str]],
        hypothesis: str,
        max_n: int = 4,
    ) -> BleuResult:
        """Compute cumulative BLEU scores for each n-gram order up to max_n, plus an overall score using uniform weights up to max_n.

        Args:
            references: The reference sentence(s). Can be a single string or
                        a list/tuple of strings.
            hypothesis: The hypothesis sentence.
            max_n: The maximum n-gram size to evaluate (default 4).

        Returns:
            A BleuResult containing the overall score (uniform weights up to max_n)
            and a dictionary `cumulative_ngram_scores` mapping n (1 to max_n)
            to the cumulative BLEU-n score.

        """  # noqa: E501
        if isinstance(references, str):
            reference_list = [references]
        elif isinstance(references, (list, tuple)):
            reference_list = references
        else:
            msg = "References must be a string or a sequence of strings."
            raise TypeError(msg)

        if not hypothesis or not reference_list or not any(reference_list):
            logging.warning("Cannot compute BLEU score with empty reference(s) or hypothesis.")  # noqa: LOG015
            return BleuResult(score=0.0, cumulative_ngram_scores=dict.fromkeys(range(1, max_n + 1), 0.0))

        try:
            # Preprocess hypothesis
            hyp_tokens: List[str] = list(self._preprocess_text(hypothesis))

            # Preprocess reference(s)
            ref_tokens_list: List[List[str]] = [list(self._preprocess_text(ref)) for ref in reference_list]

            # Handle cases where preprocessing results in empty lists
            if not hyp_tokens or not any(ref_tokens_list):
                if not hyp_tokens:
                    logging.warning(f"Hypothesis '{hypothesis[:50]}...' became empty after preprocessing.")  # noqa: G004, LOG015
                if not any(ref_tokens_list):
                    logging.warning("All references became empty after preprocessing.")  # noqa: LOG015
                return BleuResult(score=0.0, cumulative_ngram_scores=dict.fromkeys(range(1, max_n + 1), 0.0))

            cumulative_scores: Dict[int, float] = {}
            for n in range(1, max_n + 1):
                # Weights for cumulative BLEU-n: uniform up to n, zero beyond
                weights = tuple(1.0 / n if i < n else 0.0 for i in range(max_n))
                ngram_score = sentence_bleu(
                    ref_tokens_list,
                    hyp_tokens,
                    weights=weights,
                    smoothing_function=self.smoothing,
                )
                cumulative_scores[n] = ngram_score  # type: ignore # No need for float() cast  # noqa: PGH003

            # Calculate overall score using uniform weights across all considered n-grams
            uniform_weights = tuple(1.0 / max_n for _ in range(max_n))
            overall_score = sentence_bleu(
                ref_tokens_list,
                hyp_tokens,
                weights=uniform_weights,
                smoothing_function=self.smoothing,
            )

            return BleuResult(score=overall_score, cumulative_ngram_scores=cumulative_scores)  # type: ignore  # noqa: PGH003

        except Exception as e:
            ref_repr = (
                f"'{reference_list[0][:50]}...'"
                if isinstance(reference_list, list) and reference_list
                else str(reference_list)
            )
            hyp_repr = f"'{hypothesis[:50]}...'"
            logging.exception(
                f"Error computing n-gram BLEU scores for refs: {ref_repr} and hyp: {hyp_repr}. Max_n: {max_n}",  # noqa: G004
                exc_info=e,
            )
            # Return 0 scores on failure
            return BleuResult(score=0.0, cumulative_ngram_scores=dict.fromkeys(range(1, max_n + 1), 0.0))


def extract_string_similarity_vector(original: str, compare_text: str) -> dict[str, float]:
    """Extract various string similarity metrics between two texts."""
    s1 = original.lower()
    s2 = compare_text.lower()
    result = {}

    # Basic similarity metrics
    try:
        seq = difflib.SequenceMatcher(None, s1, s2)
        result.update(
            {
                "normalized_levenshtein": normalized_levenshtein.similarity(s1, s2),
                "jaro_winkler": jaro_winkler.similarity(s1, s2),
                # "rfuzz_jaro_similarity": rapidfuzz.distance.JaroWinkler.distance(s1, s2),
                "metric_lcs_similarity": 1 - metric_lcs.distance(s1, s2),
                "qgram2_similarity": qgram2.distance(s1, s2),
                "qgram3_similarity": qgram3.distance(s1, s2),
                "qgram4_similarity": qgram4.distance(s1, s2),
                "jaccard_char_2gram": jaccard.similarity(s1, s2),
                "cosine_char_2gram": sim_cosine.similarity(s1, s2),
                "rfuzz_partial_ratio": rapidfuzz.fuzz.partial_ratio(s1, s2) / 100.0,
                "rfuzz_partial_token_set_ratio": rapidfuzz.fuzz.partial_token_set_ratio(s1, s2) / 100.0,
                "rfuzz_partial_token_sort_ratio": rapidfuzz.fuzz.partial_token_sort_ratio(s1, s2) / 100.0,
                "rfuzz_token_set_ratio": rapidfuzz.fuzz.token_set_ratio(s1, s2) / 100.0,
                "rfuzz_token_sort_ratio": rapidfuzz.fuzz.token_sort_ratio(s1, s2) / 100.0,
                "rfuzz_qratio": rapidfuzz.fuzz.QRatio(s1, s2) / 100.0,
                "rfuzz_ratio": rapidfuzz.fuzz.ratio(s1, s2) / 100.0,
                "rfuzz_wratio": rapidfuzz.fuzz.WRatio(s1, s2) / 100.0,
                "fz_uqratio": fuzzy.UQRatio(s1, s2) / 100.0,
                "fz_uwratio": fuzzy.UWRatio(s1, s2) / 100.0,
                "ratio": seq.ratio(),
                "quick_ratio": seq.quick_ratio(),
                "real_quick_ratio": seq.real_quick_ratio(),
            },
        )
    except Exception:  # noqa: S110
        pass  # Optionally log the error

    # BLEU Score
    try:
        scorer = BleuScorer()
        result["bleu_score"] = scorer.score_all_ngrams(s1, s2).score
    except Exception:
        result["bleu_score"] = None

    # BM25 Score
    try:
        bm25 = BM25([s1.split()])
        tokenized_query = s2.split()
        doc_scores = bm25.get_scores(tokenized_query)
        result["bm25"] = doc_scores[0] if doc_scores else None
    except Exception:
        result["bm25"] = None

    # TF-IDF and vector distance metrics
    try:
        tfidf = TFIDF(s1, s2)
        result.update(
            {
                "tfidf_cosine_similarity": tfidf.calculate_distance("cosine"),
                "tfidf_euclidean_distance": tfidf.calculate_distance("euclidean"),
                "tfidf_manhattan_distance": tfidf.calculate_distance("manhattan"),
                "tfidf_minkowski_distance": tfidf.calculate_distance("minkowski"),
                "tfidf_jaccard_similarity": tfidf.calculate_distance("jaccard"),
                "tfidf_hamming_distance": tfidf.calculate_distance("hamming"),
            },
        )
    except Exception:
        result.update(
            {
                "tfidf_cosine_similarity": None,
                "tfidf_euclidean_distance": None,
                "tfidf_manhattan_distance": None,
                "tfidf_minkowski_distance": None,
                "tfidf_jaccard_similarity": None,
                "tfidf_hamming_distance": None,
            },
        )

    return result


if __name__ == "__main__":
    # Example usage
    original_text = "The quick brown fox jumps over the lazy dog."
    compare_text = "A fast brown fox leaped over the sleepy dog"
    similarity_vector = extract_string_similarity_vector(original_text, compare_text)
    # Output: A dictionary containing various similarity metrics and their scores
    # Note: The output will vary based on the input texts and the similarity metrics used.
    # You can adjust the original_text and compare_text variables to test different cases.
    print("\nSimilarity (Original vs. Similar):")
    for k in sorted(similarity_vector.keys()):
        v = similarity_vector[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
