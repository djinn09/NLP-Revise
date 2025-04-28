# Use annotations for cleaner type hinting (requires Python 3.7+)
from __future__ import annotations

import difflib
import logging
import math
import re
import string
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

# Attempt NLTK imports and provide guidance if missing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
except ImportError:
    msg = "NLTK library not found. Please install it: pip install nltk"
    raise ImportError(msg) from None

# Other core libraries
from rapidfuzz import fuzz as rapidfuzz_fuzz  # Alias for clarity
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# Optional libraries (handle import errors gracefully)
try:
    from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz  # Alias for clarity

    _fuzzywuzzy_available = True
except ImportError:
    _fuzzywuzzy_available = False
    warnings.warn(
        "fuzzywuzzy library not found. Some metrics (UQRatio, UWRatio) will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

try:
    from rank_bm25 import BM25L as BM25

    _bm25_available = True
except ImportError:
    _bm25_available = False
    warnings.warn("rank_bm25 library not found. BM25 metric will be unavailable.", ImportWarning, stacklevel=2)

# Similarity library components (assuming installed)
from similarity.cosine import Cosine
from similarity.jaccard import Jaccard
from similarity.jarowinkler import JaroWinkler
from similarity.metric_lcs import MetricLCS
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.qgram import QGram

# --- NLTK Data Handling ---

_NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "wordnet": "corpora/wordnet.zip/wordnet/",  # Adjusted path for direct find
    "omw-1.4": "corpora/omw-1.4.zip/omw-1.4/",  # Specific version often needed by wordnet
    "stopwords": "corpora/stopwords",
}
_NLTK_DATA_DOWNLOADED = dict.fromkeys(_NLTK_RESOURCES, False)


def _ensure_nltk_data(resource_name: str, download_dir: Optional[str] = None) -> bool:
    """Check if NLTK resource is available, downloads if not."""
    if resource_name not in _NLTK_RESOURCES:
        warnings.warn(f"Attempting to ensure unknown NLTK resource: {resource_name}", RuntimeWarning, stacklevel=2)
        return False
    if _NLTK_DATA_DOWNLOADED.get(resource_name, False):
        return True

    try:
        # Use find with adjusted path for zipped corpora if needed
        find_path = _NLTK_RESOURCES[resource_name]
        nltk.data.find(find_path)
        _NLTK_DATA_DOWNLOADED[resource_name] = True
        logging.info("NLTK data '%s' found.", resource_name)
    except LookupError:
        logging.info(f"NLTK data '{resource_name}' not found. Downloading...")
        try:
            # Use the resource name (e.g., 'omw-1.4') for download
            nltk.download(resource_name, download_dir=download_dir, quiet=True)
            # Verify download by finding again
            nltk.data.find(_NLTK_RESOURCES[resource_name])
            _NLTK_DATA_DOWNLOADED[resource_name] = True
            logging.info(f"NLTK data '{resource_name}' downloaded successfully.")
            return True
        except Exception as e:
            warnings.warn(
                f"Failed to download or verify NLTK data '{resource_name}'. Dependent features might fail. Error: {e}",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
    else:
        return True


# --- Preprocessing Setup ---
@lru_cache(maxsize=1)
def get_default_stopwords() -> Set[str]:
    """Lazily loads default English stopwords."""
    return set(stopwords.words("english")) if _ensure_nltk_data("stopwords") else set()  # Fallback


@lru_cache(maxsize=1)
def get_default_lemmatizer() -> WordNetLemmatizer:
    """Lazily loads default WordNetLemmatizer."""
    if _ensure_nltk_data("wordnet") and _ensure_nltk_data("omw-1.4"):
        result = WordNetLemmatizer()
    else:
        warnings.warn(
            "WordNet/OMW data not found or failed to download. Lemmatization might not work correctly.",
            RuntimeWarning,
            stacklevel=2,
        )
        result = WordNetLemmatizer()
    return result


@lru_cache(maxsize=1)
def get_default_stemmer() -> PorterStemmer:
    """Lazily loads default PorterStemmer."""
    return PorterStemmer()


DEFAULT_STOP_WORDS: Optional[Set[str]] = get_default_stopwords()
DEFAULT_LEMMATIZER: Optional[WordNetLemmatizer] = get_default_lemmatizer()
DEFAULT_STEMMER: Optional[PorterStemmer] = get_default_stemmer()


# Pre-compile regex patterns
SPECIAL_CHARS_REMOVE_PATTERN = re.compile(r'[-()"#/@&^*();:<>{}`+=~|!?,]')
# Pattern to keep only alphanumeric and essential intra-word characters (like hyphens if desired)
# Simpler version: keep word chars (letters, numbers, underscore) and space
TOKENIZATION_PATTERN = re.compile(r"[^\w\s]+")
# Punctuation removal map for simpler methods
REMOVE_PUNCTUATION_MAP = str.maketrans("", "", string.punctuation)

# --- Cached Preprocessing Functions ---


@lru_cache(maxsize=1024)
def preprocess_text_base(text: str) -> str:
    """Lowercase and remove punctuation."""
    if not isinstance(text, str):
        return ""  # Handle non-string input gracefully
    return text.lower().translate(REMOVE_PUNCTUATION_MAP)


@lru_cache(maxsize=1024)
def tokenize_text(text: str) -> Tuple[str, ...]:
    """Lowercase, remove punctuation, and tokenize using NLTK."""
    if not isinstance(text, str):
        return ()
    # Ensure punkt is available
    _ensure_nltk_data("punkt")
    try:
        # Use simple map for punctuation before tokenizing
        cleaned_text = text.lower().translate(REMOVE_PUNCTUATION_MAP)
        return tuple(word_tokenize(cleaned_text))
    except Exception:
        logging.exception(f"NLTK word_tokenize failed for text: '{text[:50]}...'.")
        # Fallback to simple split
        return tuple(cleaned_text.split())


@lru_cache(maxsize=1024)
def lemmatize_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    """Lemmatize a tuple of tokens."""
    lemmatizer = get_default_lemmatizer()
    try:
        return tuple(lemmatizer.lemmatize(token) for token in tokens)
    except Exception:
        # WordNet data might be missing even if instance created
        logging.exception("Lemmatization failed. Ensure WordNet/OMW data is downloaded.")
        return tokens  # Return original tokens on failure


@lru_cache(maxsize=1024)
def stem_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    """Stem a tuple of tokens."""
    stemmer = get_default_stemmer()
    return tuple(stemmer.stem(token) for token in tokens)


@lru_cache(maxsize=1024)
def filter_stopwords(tokens: Tuple[str, ...], stop_words: Optional[frozenset[str]] = None) -> Tuple[str, ...]:
    """Filter stopwords from a tuple of tokens."""
    sw = stop_words or frozenset(get_default_stopwords())
    return tuple(token for token in tokens if token not in sw and token.isalnum())


# --- Similarity Metric Instances (Globally Initialized) ---
# These are generally stateless or thread-safe

NORMALIZED_LEVENSHTEIN = NormalizedLevenshtein()
JARO_WINKLER = JaroWinkler()
METRIC_LCS = MetricLCS()
QGRAM_2 = QGram(2)
QGRAM_3 = QGram(3)
QGRAM_4 = QGram(4)
SIM_COSINE = Cosine(2)  # Cosine similarity on character 2-grams
SIM_JACCARD = Jaccard(2)  # Jaccard similarity on character 2-grams


# --- TFIDF Class (Refactored) ---
@dataclass
class TfidfConfig:
    """Configuration for the TfidfVectorizer.

    Attributes:
        token_pattern (str): Regular expression for tokenization.
        ngram_range (Tuple[int, int]): The range of n-grams to consider.
        max_df (float): Maximum document frequency for filtering terms.
        min_df (int): Minimum document frequency for filtering terms.

    """

    token_pattern: str = r"(?u)\b\w\w+\b"  # noqa: S105
    ngram_range: Tuple[int, int] = (1, 1)
    max_df: float = 1.0
    min_df: int = 1


class TFIDFCalculator:
    """Computes TF-IDF vectors and derived similarity/distance metrics.

    Designed to be instantiated once with configuration and reused.
    The `fit_transform` method should be called for each pair or batch of texts.
    """

    def __init__(
        self,
        *,
        use_lemmatization: bool = True,
        use_stopwords: bool = True,
        stop_words: Optional[Set[str]] = None,
        tfidf_config: Optional[TfidfConfig] = None,
        **tfidf_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize the TFIDFCalculator with configuration options.

        Args:
            use_lemmatization (bool): Whether to use lemmatization.
            use_stopwords (bool): Whether to use stopwords.
            stop_words (Optional[Set[str]]): Custom stopwords to use.
            tfidf_config (TfidfConfig): Configuration for TfidfVectorizer.
            **tfidf_kwargs: Additional keyword arguments for TfidfVectorizer.

        """
        self.use_lemmatization = use_lemmatization
        self.use_stopwords = use_stopwords
        self.custom_stop_words = frozenset(stop_words) if stop_words else frozenset(get_default_stopwords())
        self._tokenizer = self._build_tokenizer()

        # Initialize TfidfVectorizer
        tfidf_config = tfidf_config or TfidfConfig()
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenizer if self._tokenizer else None,
            token_pattern=tfidf_config.token_pattern if not self._tokenizer else "",
            stop_words=list(self.custom_stop_words)
            if use_stopwords and not self._tokenizer
            else None,  # Sklearn handles stopwords if no custom tokenizer
            ngram_range=tfidf_config.ngram_range,
            max_df=tfidf_config.max_df,
            min_df=tfidf_config.min_df,
            **tfidf_kwargs,
        )
        logging.info(f"TFIDFCalculator initialized. Lemmatization: {use_lemmatization}, Stopwords: {use_stopwords}")

    def _build_tokenizer(self) -> Optional[Callable[[str], List[str]]]:
        """Create a tokenizer function based on normalization settings."""
        if (
            self.use_lemmatization or self.use_stopwords
        ):  # Need custom handling if lemmatizing or using specific stop-word logic
            sw = self.custom_stop_words if self.use_stopwords else frozenset()

            def tokenizer_func(text: str) -> List[str]:
                tokens = tokenize_text(text)  # Base tokenization (cached)
                if self.use_lemmatization:
                    tokens = lemmatize_tokens(tokens)  # Lemmatize (cached)
                if self.use_stopwords:
                    # Use isalnum filtering here as well
                    tokens = tuple(t for t in tokens if t not in sw and t.isalnum())
                return list(tokens)

            return tokenizer_func
        # Let TfidfVectorizer handle tokenization with token_pattern if no custom norm needed
        return None

    def fit_transform(self, texts: Sequence[str]) -> csr_matrix:
        """Fits the vectorizer and transforms the input texts."""
        try:
            return csr_matrix(self.vectorizer.fit_transform(texts))
        except Exception:
            logging.exception("TF-IDF fit_transform failed")
            # Return an empty sparse matrix matching the expected shape
            return csr_matrix((len(texts), 0), dtype=float)

    def calculate_metrics_pairwise(self, text1: str, text2: str) -> Dict[str, Optional[float]]:
        """Calculate various TF-IDF-based similarity and distance metrics for a pair of texts.

        Args:
            text1 (str): The first text input.
            text2 (str): The second text input.

        Returns:
            Dict[str, Optional[float]]: A dictionary containing similarity and distance metrics,
            such as cosine similarity, Euclidean distance, and Jaccard similarity.

        """
        metrics = {
            "tfidf_cosine_similarity": None,
            "tfidf_euclidean_distance": None,
            "tfidf_manhattan_distance": None,
            "tfidf_jaccard_similarity": None,  # Jaccard on binary presence
            "tfidf_hamming_distance": None,  # Hamming on binary presence
        }
        try:
            tfidf_matrix = self.fit_transform([text1, text2])

            # Handle case where TF-IDF fails or results in empty matrix
            if tfidf_matrix.shape[1] == 0:  # No features found
                logging.warning(f"TF-IDF found no features for texts: '{text1[:50]}...', '{text2[:50]}...'")
                # Return 0 similarity / max distance (or keep None) - let's default to 0 sim / inf dist
                metrics["tfidf_cosine_similarity"] = 0.0
                metrics["tfidf_jaccard_similarity"] = 0.0
                # For distances, infinity might be more appropriate, or None
                metrics["tfidf_euclidean_distance"] = float("inf")
                metrics["tfidf_manhattan_distance"] = float("inf")
                metrics["tfidf_hamming_distance"] = 1.0  # Max possible hamming distance
                return metrics

            # 1. Cosine Similarity
            sim_matrix = cosine_similarity(tfidf_matrix)
            metrics["tfidf_cosine_similarity"] = float(sim_matrix[0, 1])

            # 2. Distances (Euclidean, Manhattan)
            # pairwise_distances needs dense arrays for some metrics
            dense_matrix = tfidf_matrix.toarray()
            metrics["tfidf_euclidean_distance"] = float(pairwise_distances(dense_matrix, metric="euclidean")[0, 1])
            metrics["tfidf_manhattan_distance"] = float(pairwise_distances(dense_matrix, metric="manhattan")[0, 1])
            metrics["tfidf_minkowski_distance"] = float(pairwise_distances(dense_matrix, metric="minkowski")[0, 1])

            # 3. Jaccard & Hamming on Binarized Vectors
            # Convert TF-IDF to binary presence (term exists or not)
            binary_presence = (dense_matrix > 0).astype(bool)

            # Check if vectors are non-zero before calculating Jaccard/Hamming
            if binary_presence[0].any() or binary_presence[1].any():
                # Jaccard Similarity (1 - Jaccard Distance)
                # sklearn pairwise_distances 'jaccard' returns distance

                # Summarizing my current belief:
                # - Micro-Jaccard: good if you want a "how much vocab overlaps" score.
                # - Weighted-Jaccard: better for longer texts where important words matter more.
                # - Macro-Jaccard: risky and too noisy.
                # - Binary-Jaccard: acceptable if TF-IDF already filtered out unimportant words.
                jaccard_dist = pairwise_distances(binary_presence, metric="jaccard")[0, 1]
                # Handle potential division by zero if both vectors are all zeros (though handled above)
                metrics["tfidf_jaccard_similarity"] = float(jaccard_dist) if not math.isnan(jaccard_dist) else 0.0

                # Hamming Distance (fraction of positions differing)
                # sklearn pairwise_distances 'hamming' calculates this directly
                hamming_dist = pairwise_distances(binary_presence.astype(int), metric="hamming")[0, 1]
                metrics["tfidf_hamming_distance"] = float(hamming_dist)
            else:
                # If both binary vectors are all zeros
                metrics["tfidf_jaccard_similarity"] = 1.0  # Identical (both empty)
                metrics["tfidf_hamming_distance"] = 0.0  # Identical (no differing bits)

        except Exception as e:
            logging.exception(f"Error calculating TF-IDF metrics for '{text1[:50]}...' vs '{text2[:50]}...': {e}")
            # Leave metrics as None on failure

        return metrics


# --- BleuScorer Class (Using Refined Version) ---
# (Assuming the improved BleuScorer from previous interaction is available or paste it here)
# For brevity, I'll reuse the essential parts here, adapted slightly


@dataclass
class BleuResult:
    """Holds BLEU scoring results."""

    score: float  # Overall score (e.g., uniform BLEU-4)
    cumulative_ngram_scores: Optional[Dict[int, float]] = field(default=None)


class BleuScorer:
    """Computes BLEU similarity (adapted)."""

    def __init__(
        self,
        stop_words: Optional[Set[str]] = None,
        lemmatizer: Optional[WordNetLemmatizer] = None,
        smoothing_function: Optional[Callable] = None,
    ):
        self.lemmatizer = lemmatizer or get_default_lemmatizer()
        self.stop_words = frozenset(stop_words or get_default_stopwords())  # Use frozenset for caching
        self.smoothing = smoothing_function or SmoothingFunction().method1
        logging.info("BleuScorer initialized.")

    @lru_cache(maxsize=1024)  # Cache preprocessing based on text and stop_words
    def _preprocess_bleu_text(self, text: str, current_stopwords: frozenset[str]) -> Tuple[str, ...]:
        """Preprocesses text specifically for BLEU."""
        try:
            _ensure_nltk_data("punkt")  # Ensure tokenizer data
            # BLEU often benefits from keeping case, but removing punctuation
            # Let's stick to lowercase + basic punctuation removal for consistency here
            # More advanced recipes exist (e.g., Moses tokenizer scripts)
            cleaned = text.lower().translate(REMOVE_PUNCTUATION_MAP)
            tokens = word_tokenize(cleaned)
            processed = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token and token not in current_stopwords  # Filter empty and stop words
            ]
            return tuple(processed)
        except Exception as e:
            logging.exception(f"BLEU preprocessing failed for '{text[:50]}...': {e}")
            return ()

    def _calculate_bleu(
        self, ref_tokens_list: List[List[str]], hyp_tokens: List[str], weights: Tuple[float, ...]
    ) -> float:
        """Internal BLEU calculation with error handling."""
        if not hyp_tokens or not any(ref_tokens_list):
            return 0.0
        try:
            # Note: NLTK's sentence_bleu expects list of lists for references
            return sentence_bleu(
                references=ref_tokens_list, hypothesis=hyp_tokens, weights=weights, smoothing_function=self.smoothing
            )
        except ZeroDivisionError:
            # Can happen with very short sentences and no smoothing
            logging.warning(
                f"BLEU calculation resulted in ZeroDivisionError (likely short hypothesis). Returning 0.0. Hyp: {hyp_tokens}"
            )
            return 0.0
        except Exception as e:
            logging.exception(f"Unexpected error during sentence_bleu calculation: {e}")
            return 0.0

    def score(
        self,
        references: Union[str, Sequence[str]],
        hypothesis: str,
        weights: Tuple[float, ...] = (0.25, 0.25, 0.25, 0.25),  # Default BLEU-4
    ) -> BleuResult:
        """Computes BLEU score."""
        ref_list = [references] if isinstance(references, str) else references
        if not hypothesis or not ref_list or not any(ref_list):
            return BleuResult(score=0.0)

        # Use the instance's stop_words (as frozenset) for caching
        hyp_tokens = list(self._preprocess_bleu_text(hypothesis, self.stop_words))
        ref_tokens_list = [list(self._preprocess_bleu_text(ref, self.stop_words)) for ref in ref_list]

        score_value = self._calculate_bleu(ref_tokens_list, hyp_tokens, weights)
        return BleuResult(score=score_value)

    def score_all_ngrams(self, references: Union[str, Sequence[str]], hypothesis: str, max_n: int = 4) -> BleuResult:
        """Computes cumulative BLEU-1 to BLEU-N scores."""
        ref_list = [references] if isinstance(references, str) else references
        if not hypothesis or not ref_list or not any(ref_list):
            return BleuResult(score=0.0, cumulative_ngram_scores={n: 0.0 for n in range(1, max_n + 1)})

        hyp_tokens = list(self._preprocess_bleu_text(hypothesis, self.stop_words))
        ref_tokens_list = [list(self._preprocess_bleu_text(ref, self.stop_words)) for ref in ref_list]

        cumulative_scores: Dict[int, float] = {}
        for n in range(1, max_n + 1):
            weights = tuple(1.0 / n if i < n else 0.0 for i in range(max_n))  # Standard cumulative weights
            ngram_score = self._calculate_bleu(ref_tokens_list, hyp_tokens, weights)
            cumulative_scores[n] = ngram_score

        # Overall score often uses uniform weights up to max_n
        uniform_weights = tuple(1.0 / max_n for _ in range(max_n))
        overall_score = self._calculate_bleu(ref_tokens_list, hyp_tokens, uniform_weights)

        return BleuResult(score=overall_score, cumulative_ngram_scores=cumulative_scores)


# --- BM25 Calculation Wrapper ---
def calculate_bm25(reference: str, hypothesis: str) -> Optional[float]:
    """Calculates BM25 score. Returns None if library unavailable or error."""
    if not _bm25_available:
        return None
    try:
        # BM25 expects a corpus of *tokenized* documents
        # Using simple split here, could use cached tokenize_text
        tokenized_corpus = [list(tokenize_text(reference))]
        tokenized_query = list(tokenize_text(hypothesis))

        if not tokenized_corpus or not tokenized_corpus[0] or not tokenized_query:
            return 0.0  # Score is 0 if query or document is empty after tokenization

        bm25 = BM25(tokenized_corpus)
        doc_scores = bm25.get_scores(tokenized_query)
        return doc_scores[0] if doc_scores else 0.0
    except Exception as e:
        logging.error(f"BM25 calculation failed for '{reference[:50]}...' vs '{hypothesis[:50]}...': {e}")
        return None


# --- Main Similarity Calculator Class ---


class SimilarityCalculator:
    """Orchestrates calculation of various text similarity metrics."""

    def __init__(
        self,
        use_lemmatization: bool = True,
        use_stopwords: bool = True,
        custom_stop_words: Optional[Set[str]] = None,
        tfidf_options: Optional[Dict[str, Any]] = None,
        bleu_smoothing_function: Optional[Callable] = None,
    ):
        """Initialize the calculator with shared configurations."""
        logging.info("Initializing SimilarityCalculator...")
        # Ensure base NLTK data needed for defaults
        _ensure_nltk_data("punkt")
        _ensure_nltk_data("stopwords")
        if use_lemmatization:
            _ensure_nltk_data("wordnet")
            _ensure_nltk_data("omw-1.4")

        self.use_lemmatization = use_lemmatization
        self.use_stopwords = use_stopwords
        self.stop_words = custom_stop_words or get_default_stopwords()
        self.lemmatizer = get_default_lemmatizer() if use_lemmatization else None

        # Instantiate reusable components
        self.bleu_scorer = BleuScorer(
            stop_words=self.stop_words, lemmatizer=self.lemmatizer, smoothing_function=bleu_smoothing_function
        )
        self.tfidf_calculator = TFIDFCalculator(
            use_lemmatization=use_lemmatization,
            use_stopwords=use_stopwords,
            stop_words=self.stop_words,
            **(tfidf_options or {}),
        )
        logging.info("SimilarityCalculator initialized successfully.")

    def calculate_single_pair(self, text1: str, text2: str) -> Dict[str, Optional[float]]:
        """Calculate all configured similarity metrics for a single pair of texts."""
        if not isinstance(text1, str) or not isinstance(text2, str):
            logging.warning(f"Invalid input types for similarity calculation: {type(text1)}, {type(text2)}")
            return {}  # Return empty dict for invalid input

        results: Dict[str, Optional[float]] = {}

        # 1. Basic String / Sequence Metrics (operate on raw or lightly preprocessed)
        s1_lower = text1.lower()
        s2_lower = text2.lower()
        try:
            seq = difflib.SequenceMatcher(None, s1_lower, s2_lower, autojunk=False)  # autojunk=False for accuracy
            results["ratio"] = seq.ratio()
            results["quick_ratio"] = seq.quick_ratio()
            results["real_quick_ratio"] = seq.real_quick_ratio()

            results["normalized_levenshtein"] = NORMALIZED_LEVENSHTEIN.similarity(s1_lower, s2_lower)
            results["jaro_winkler"] = JARO_WINKLER.similarity(s1_lower, s2_lower)
            results["metric_lcs_similarity"] = 1.0 - METRIC_LCS.distance(
                s1_lower,
                s2_lower,
            )  # similarity = 1 - distance
            results["qgram2_similarity"] = QGRAM_2.distance(s1_lower, s2_lower)
            results["qgram3_similarity"] = QGRAM_3.distance(s1_lower, s2_lower)
            results["qgram4_similarity"] = QGRAM_4.distance(s1_lower, s2_lower)
            results["cosine_char_2gram"] = SIM_COSINE.similarity(s1_lower, s2_lower)
            results["jaccard_char_2gram"] = SIM_JACCARD.similarity(s1_lower, s2_lower)

        except Exception as e:
            logging.exception(f"Error during basic string similarity calculation: {e}")
            # Set potentially calculated values to None on error in this block
            for k in [
                "ratio",
                "quick_ratio",
                "real_quick_ratio",
                "normalized_levenshtein",
                "jaro_winkler",
                "metric_lcs_similarity",
                "qgram2_similarity",
                "qgram3_similarity",
                "qgram4_similarity",
                "cosine_char_2gram",
                "jaccard_char_2gram",
            ]:
                results[k] = None

        # 2. RapidFuzz Metrics (optimized fuzzy matching)
        try:
            results["rfuzz_ratio"] = rapidfuzz_fuzz.ratio(s1_lower, s2_lower) / 100.0
            results["rfuzz_partial_ratio"] = rapidfuzz_fuzz.partial_ratio(s1_lower, s2_lower) / 100.0
            results["rfuzz_token_set_ratio"] = rapidfuzz_fuzz.token_set_ratio(s1_lower, s2_lower) / 100.0
            results["rfuzz_token_sort_ratio"] = rapidfuzz_fuzz.token_sort_ratio(s1_lower, s2_lower) / 100.0
            results["rfuzz_partial_token_set_ratio"] = (
                rapidfuzz_fuzz.partial_token_set_ratio(s1_lower, s2_lower) / 100.0
            )
            results["rfuzz_partial_token_sort_ratio"] = (
                rapidfuzz_fuzz.partial_token_sort_ratio(s1_lower, s2_lower) / 100.0
            )
            results["rfuzz_wratio"] = rapidfuzz_fuzz.WRatio(s1_lower, s2_lower) / 100.0
            results["rfuzz_qratio"] = rapidfuzz_fuzz.QRatio(s1_lower, s2_lower) / 100.0
            # Jaro-Winkler distance -> similarity
            # Note: rapidfuzz distance functions might need specific parameters
            # Let's use the similarity function directly if available, or calculate from distance
            # rapidfuzz.distance.JaroWinkler.distance(s1, s2) -> returns distance
            # For consistency with other libraries, let's use similarity version if possible,
            # Assuming rapidfuzz.fuzz provides similarity scores scaled 0-100
            # No direct JaroWinkler similarity in fuzz? Use distance and convert.
            # Check documentation: JaroWinkler distance seems to be edit distance, not normalized.
            # Let's stick to the similarity library's JaroWinkler for consistency above.

        except Exception as e:
            logging.exception(f"Error during RapidFuzz calculation: {e}")
            for k in [
                "rfuzz_ratio",
                "rfuzz_partial_ratio",
                "rfuzz_token_set_ratio",
                "rfuzz_token_sort_ratio",
                "rfuzz_partial_token_set_ratio",
                "rfuzz_partial_token_sort_ratio",
                "rfuzz_wratio",
                "rfuzz_qratio",
            ]:
                results[k] = None

        # 3. FuzzyWuzzy Metrics (if available)
        if _fuzzywuzzy_available:
            try:
                results["fz_uqratio"] = fuzzywuzzy_fuzz.UQRatio(s1_lower, s2_lower) / 100.0
                results["fz_uwratio"] = fuzzywuzzy_fuzz.UWRatio(s1_lower, s2_lower) / 100.0
            except Exception as e:
                logging.exception(f"Error during FuzzyWuzzy calculation: {e}")
                results["fz_uqratio"] = None
                results["fz_uwratio"] = None
        else:
            results["fz_uqratio"] = None
            results["fz_uwratio"] = None

        # 4. BLEU Score (using configured BleuScorer)
        try:
            bleu_result = self.bleu_scorer.score_all_ngrams(text1, text2)  # Use original case text for BLEU typically
            results["bleu_score"] = bleu_result.score
            # Optionally add cumulative scores
            # if bleu_result.cumulative_ngram_scores:
            #     for n, score in bleu_result.cumulative_ngram_scores.items():
            #         results[f"bleu_{n}_cumulative"] = score
        except Exception as e:
            logging.exception(f"Error calculating BLEU score: {e}")
            results["bleu_score"] = None

        # 5. BM25 Score (using wrapper)
        try:
            results["bm25"] = calculate_bm25(text1, text2)  # Pass original texts
        except Exception as e:
            logging.exception(f"Error calculating BM25 score: {e}")
            results["bm25"] = None

        # 6. TF-IDF Metrics (using configured TFIDFCalculator)
        try:
            tfidf_metrics = self.tfidf_calculator.calculate_metrics_pairwise(text1, text2)
            results.update(tfidf_metrics)
        except Exception as e:
            logging.exception(f"Error calculating TF-IDF metrics: {e}")
            for k in [
                "tfidf_cosine_similarity",
                "tfidf_euclidean_distance",
                "tfidf_manhattan_distance",
                "tfidf_jaccard_similarity",
                "tfidf_hamming_distance",
                "tfidf_minkowski_distance",
            ]:
                results[k] = None

        # Final check for None vs NaN - prefer None for consistency
        for k, v in results.items():
            if isinstance(v, float) and math.isnan(v):
                results[k] = None

        return results

    def calculate_multiple_pairs(
        self,
        text_pairs: Iterable[Tuple[str, str]],
        max_workers: Optional[int] = None,  # Number of parallel processes
    ) -> List[Dict[str, Optional[float]]]:
        """Calculate similarity metrics for multiple text pairs in parallel.

        Args:
            text_pairs: An iterable of tuples, where each tuple is (text1, text2).
            max_workers: The maximum number of worker processes to use.
                         If None, uses the default (usually number of cores).

        Returns:
            A list of dictionaries, where each dictionary contains the
            similarity metrics for the corresponding input pair. The order
            matches the input iterable order.

        """
        results = []
        text_pairs_list = list(text_pairs)  # Consume iterator if needed
        if not text_pairs_list:
            return []

        # Configuration to pass to workers - ONLY PRIMITIVE, PICKLABLE VALUES
        config = {
            "use_lemmatization": self.use_lemmatization,
            "use_stopwords": self.use_stopwords,
            # Pass stop words set (frozenset is generally picklable)
            "custom_stop_words": frozenset(self.stop_words) if self.stop_words else None,
            # Pass smoothing function (assuming it's a top-level function or picklable)
            "bleu_smoothing_function": self.bleu_scorer.smoothing,
            # --- DO NOT PASS TFIDF OPTIONS DERIVED FROM get_params() ---
            # The worker will initialize TFIDF based on the flags above.
            # If you needed *other* specific TFIDF params (like ngram_range, min_df),
            # pass them explicitly here as primitive types:
            # "tfidf_ngram_range": self.tfidf_calculator.vectorizer.ngram_range, # Example
            # "tfidf_min_df": self.tfidf_calculator.vectorizer.min_df,      # Example
        }
        # Retrieve specific TFIDF params if needed for worker initialization
        tfidf_params_to_pass = {
            "ngram_range": self.tfidf_calculator.vectorizer.ngram_range,
            "max_df": self.tfidf_calculator.vectorizer.max_df,
            "min_df": self.tfidf_calculator.vectorizer.min_df,
            # Add any other specific TfidfVectorizer params you set in the main init
        }
        config["tfidf_init_options"] = tfidf_params_to_pass  # Pass these separately

        futures = {}
        # Use try-with-resources for the executor
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                logging.info(
                    f"Submitting {len(text_pairs_list)} tasks to ProcessPoolExecutor with max_workers={executor._max_workers}"
                )
                for i, (text1, text2) in enumerate(text_pairs_list):
                    future = executor.submit(_worker_calculate_single_pair, config, text1, text2)
                    futures[future] = i

                results_unordered = {}
                # Using as_completed is generally better for processing as tasks finish
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        result = future.result()
                        results_unordered[index] = result
                    except Exception as e:
                        logging.error(
                            f"Error processing pair {index} in parallel worker: {e}", exc_info=False
                        )  # Don't log full trace for every failure
                        # Log the specific error from the worker if possible
                        logging.error(f"Worker error details for pair {index}: {e}")
                        results_unordered[index] = {}
        except Exception as e:
            # Catch errors during executor setup or shutdown
            logging.exception(f"Error occurred in ProcessPoolExecutor management: {e}")
            # Ensure results list has the correct size even if processing failed midway
            results = [results_unordered.get(i, {}) for i in range(len(text_pairs_list))]
            return results

        # Reconstruct results in the original order
        results = [results_unordered.get(i, {}) for i in range(len(text_pairs_list))]
        logging.info(f"Finished processing {len(results)} pairs in parallel.")
        return results


# --- Worker Function for Parallel Execution ---
# Must be defined at the top level or be a static method to be easily picklable.


def _worker_calculate_single_pair(config: Dict[str, Any], text1: str, text2: str) -> Dict[str, Optional[float]]:
    """Worker function to calculate similarity for a single pair. It reconstructs necessary components based on the config."""
    # Extract TFIDF specific init options
    tfidf_init_opts = config.get("tfidf_init_options", {})

    # Re-initialize components within the worker using the passed config
    calculator = SimilarityCalculator(
        use_lemmatization=config.get("use_lemmatization", True),
        use_stopwords=config.get("use_stopwords", True),
        custom_stop_words=config.get("custom_stop_words"),  # Will be None or frozenset
        # Pass explicit TFIDF options extracted from config
        tfidf_options=tfidf_init_opts,
        bleu_smoothing_function=config.get("bleu_smoothing_function"),
    )
    return calculator.calculate_single_pair(text1, text2)


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Initialize Calculator ---
    # Use default settings: lemmatization=True, stopwords=True
    calculator = SimilarityCalculator()

    # Example Texts
    original_text = "The quick brown fox jumps over the lazy dog."
    compare_text_similar = "A fast brown fox leaped over the sleepy dog"
    compare_text_different = "This sentence is quite different and shares few words."
    compare_text_short = "fox dog"
    empty_text = ""

    # --- Single Pair Calculation ---
    print("\n--- Single Pair Calculations ---")
    similarity_vector1 = calculator.calculate_single_pair(original_text, compare_text_similar)
    print("\nSimilarity (Original vs. Similar):")
    for k in sorted(similarity_vector1.keys()):
        v = similarity_vector1[k]
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # similarity_vector2 = calculator.calculate_single_pair(original_text, compare_text_different)
    # print(f"\nSimilarity (Original vs. Different):")
    # Print only a few key metrics for brevity
    # for k in [
    #     "ratio",
    #     "normalized_levenshtein",
    #     "rfuzz_token_set_ratio",
    #     "bleu_score",
    #     "tfidf_cosine_similarity",
    #     "bm25",
    # ]:
    #     v = similarity_vector2.get(k)
    #     print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # similarity_vector3 = calculator.calculate_single_pair(original_text, empty_text)
    # print(f"\nSimilarity (Original vs. Empty): {similarity_vector3}")

    # --- Parallel Calculation for Multiple Pairs ---
    # print("\n--- Parallel Multi-Pair Calculation ---")
    # text_pairs_to_process = [
    #     (original_text, compare_text_similar),
    #     (original_text, compare_text_different),
    #     (compare_text_similar, compare_text_short),
    #     ("Another example text.", "This is another example."),
    #     ("Short text", "Very short text"),
    #     (original_text, empty_text),  # Include edge case
    # ]

    # Use context manager for the executor if you only run parallel tasks once
    # Or keep the executor alive if you submit multiple batches
    # parallel_results = calculator.calculate_multiple_pairs(text_pairs_to_process, max_workers=4)  # Adjust max_workers

    # print(f"\nParallel Results ({len(parallel_results)} pairs processed):")
    # for i, result_dict in enumerate(parallel_results):
    #     pair = text_pairs_to_process[i]
    #     print(f"\nPair {i + 1}: '{pair[0][:30]}...' vs '{pair[1][:30]}...'")
    #     # Print a subset of results for brevity
    #     print(f"  ratio: {result_dict.get('ratio'):.4f}" if result_dict.get("ratio") is not None else "  ratio: None")
    #     print(
    #         f"  bleu_score: {result_dict.get('bleu_score'):.4f}"
    #         if result_dict.get("bleu_score") is not None
    #         else "  bleu_score: None"
    #     )
    #     print(
    #         f"  tfidf_cosine_similarity: {result_dict.get('tfidf_cosine_similarity'):.4f}"
    #         if result_dict.get("tfidf_cosine_similarity") is not None
    #         else "  tfidf_cosine_similarity: None"
    #     )

    # print("\n--- Finished ---")
