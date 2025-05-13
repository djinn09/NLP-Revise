"""Module for matching keywords from one paragraph to another and scoring.

Provides a KeywordMatcher class with configurable preprocessing and
keyword extraction methods (including POS tagging). Calculates two scores:
1. Keyword Coverage: Proportion of keywords from A found in B.
2. Vocabulary Cosine Similarity: Cosine similarity based on shared non-stop words.
Uses rich logging.

**IMPORTANT:** This version assumes necessary NLTK data ('punkt', 'stopwords',
'wordnet', 'omw-1.4', 'averaged_perceptron_tagger') has been manually
downloaded beforehand. Run the following in your Python environment if needed:

>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('stopwords')
>>> nltk.download('wordnet')
>>> nltk.download('omw-1.4')
>>> nltk.download('averaged_perceptron_tagger')
"""

from __future__ import annotations

import logging
import string
from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple, Union

# Attempt to import necessary libraries and provide guidance
try:
    from nltk import pos_tag
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
except ImportError as e:
    missing_lib = "nltk"
    error_message = f"Missing {missing_lib}. Please install it (e.g., `pip install {missing_lib}`). Original error: {e}"
    raise ImportError(error_message) from e

# Attempt to import rich
try:
    from rich.logging import RichHandler

    _rich_available = True
except ImportError:
    _rich_available = False
    msg = "Missing rich. Please install it (`pip install rich`) for enhanced logging."
    raise ImportError(msg) from None


# --- Global Resources ---
_LEMMA_INIT_FAILED = False
try:
    lemmatizer = WordNetLemmatizer()
    _ = lemmatizer.lemmatize("tests")  # Check if it works
except LookupError as e:
    print(f"[ERROR] NLTK LookupError initializing WordNetLemmatizer: {e}")
    print("        Ensure 'wordnet' and 'omw-1.4' NLTK data are downloaded.")
    lemmatizer = None
    _LEMMA_INIT_FAILED = True
except Exception as e:
    print(f"[ERROR] Unexpected error initializing WordNetLemmatizer: {e}")
    lemmatizer = None
    _LEMMA_INIT_FAILED = True

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
DEFAULT_ALLOWED_POS_TAGS = {"NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS"}
logger = logging.getLogger(__name__)
GOOD_KEYWORD_COVERAGE = 0.5
GOOD_VOCAB_COSINE = 0.5
BAD_VOCAB_COSINE = 0.1


class KeywordMatcher:
    """Matches keywords from paragraph A in paragraph B and provides scores.

    Calculates two scores:
    1. Keyword Coverage: Proportion of unique keywords extracted from A
       (based on config) found in processed B.
    2. Vocabulary Cosine Similarity: Cosine similarity of binary vectors
       representing the presence/absence of non-stop words in A and B.

    **Note:** Requires NLTK data ('punkt', 'stopwords', 'wordnet', 'omw-1.4',
    'averaged_perceptron_tagger') to be pre-downloaded. See module docstring.

    Args:
        use_lemmatization: If True, attempt to lemmatize words during keyword
                           extraction and paragraph B normalization for coverage score.
                           Defaults to True. Will be disabled if lemmatizer failed init.
                           *Note: Lemmatization is NOT used for vocabulary cosine score.*
        use_pos_tagging: If True, extract keywords for coverage score based on
                         allowed POS tags from paragraph_a. Defaults to False.
        allowed_pos_tags: A set of NLTK POS tags to consider as keywords if
                          `use_pos_tagging` is True. Defaults to nouns & adjectives.
        custom_stop_words: An optional set of custom stop words to add to the
                           default NLTK English list.

    """

    def __init__(
        self,
        *,
        use_lemmatization: bool = True,
        use_pos_tagging: bool = False,
        allowed_pos_tags: Optional[Set[str]] = None,
        custom_stop_words: Optional[Set[str]] = None,
    ) -> None:
        # --- Initial Warning ---
        logger.warning(
            "[bold yellow]Initializing KeywordMatcher. Ensure required NLTK data is downloaded![/bold yellow]",
        )

        # --- Validate Configuration ---
        if use_pos_tagging and not allowed_pos_tags:
            logger.info("POS tagging enabled, using default allowed_pos_tags (nouns & adjectives).")
            self.allowed_pos_tags = DEFAULT_ALLOWED_POS_TAGS
        elif use_pos_tagging and allowed_pos_tags:
            self.allowed_pos_tags = allowed_pos_tags
        else:
            self.allowed_pos_tags = None

        self.use_lemmatization = use_lemmatization and not _LEMMA_INIT_FAILED
        if use_lemmatization and _LEMMA_INIT_FAILED:
            logger.error(
                "Lemmatization requested, but lemmatizer failed to initialize. Lemmatization is DISABLED for coverage score.",
            )

        self.use_pos_tagging = use_pos_tagging

        # --- Setup Stop Words ---
        try:
            nltk_stopwords = set(stopwords.words("english"))
            self._stopwords_loaded = True
        except LookupError:
            logger.exception(
                "Failed to load NLTK stopwords (data likely missing 'stopwords'). Proceeding without NLTK stopwords.",
            )
            nltk_stopwords = set()
            self._stopwords_loaded = False
        except Exception as e:
            logger.exception(f"An unexpected error occurred loading stopwords: {e}")
            nltk_stopwords = set()
            self._stopwords_loaded = False

        self.stop_words = nltk_stopwords.union(custom_stop_words or set())
        if not self.stop_words:
            logger.warning("No stopwords defined.")

        # --- Log Initialization ---
        logger.info(
            f"KeywordMatcher initialized. Lemmatization (for coverage): {self.use_lemmatization}, "
            f"POS Tagging (for coverage): {self.use_pos_tagging}, "
            f"NLTK Stopwords loaded: {self._stopwords_loaded}. "
            f"{('Allowed POS: ' + str(self.allowed_pos_tags)) if self.use_pos_tagging else ''}",
        )

    @lru_cache(maxsize=128)
    def _preprocess_text(self, text: str) -> List[str]:
        """Lowercase, remove punctuation, tokenize, and remove stopwords."""
        if not isinstance(text, str) or not text.strip():
            logger.debug("Preprocessing empty or invalid text. Returning empty list.")
            return []
        try:
            cleaned_text = text.lower().translate(PUNCTUATION_TABLE)
            tokens = word_tokenize(cleaned_text)
            processed_tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
            logger.debug(f"Preprocessing result for '{text[:30]}...': {len(processed_tokens)} tokens.")
            return processed_tokens
        except LookupError:
            logger.exception(
                "NLTK LookupError during tokenization (likely missing 'punkt' data). Returning empty token list.",
            )
            return []
        except Exception:
            logger.exception(f"Unexpected error during basic preprocessing of text: '{text[:50]}...'")
            return []

    @lru_cache(maxsize=128)
    def _normalize_tokens(self, tokens: Tuple[str, ...]) -> Tuple[str, ...]:
        """Apply lemmatization (if enabled and available) to a tuple of tokens."""
        if not self.use_lemmatization:
            return tokens
        if lemmatizer is None:
            return tokens
        try:
            normalized = tuple(lemmatizer.lemmatize(token) for token in tokens)
            logger.debug(f"Lemmatized {len(tokens)} tokens.")
            return normalized
        except LookupError:
            logger.exception(
                "NLTK LookupError during lemmatization (likely missing 'wordnet'/'omw-1.4'). Returning un-normalized tokens.",
            )
            return tokens
        except Exception:
            logger.exception(f"Unexpected error during lemmatization of {len(tokens)} tokens.")
            return tokens

    @lru_cache(maxsize=128)
    def _get_pos_tags(self, tokens: Tuple[str, ...]) -> List[Tuple[str, str]]:
        """Get Part-of-Speech tags for a tuple of tokens."""
        if not tokens:
            return []
        try:
            tags = pos_tag(tokens)
            logger.debug(f"POS tagged {len(tokens)} tokens.")
            return tags
        except LookupError:
            logger.exception(
                "NLTK LookupError during POS tagging (likely missing 'averaged_perceptron_tagger').Cannot perform POS tagging.",
            )
            return []
        except Exception:
            logger.exception(f"Unexpected error during POS tagging of {len(tokens)} tokens.")
            return []

    def _extract_keywords_from_a(self, paragraph_a: str) -> Set[str]:
        """Extract keywords from paragraph_a based on configuration (for coverage score)."""
        processed_tokens = self._preprocess_text(paragraph_a)
        if not processed_tokens:
            logger.warning("Keyword extraction failed: Preprocessing returned no tokens.")
            return set()

        processed_tokens_tuple = tuple(processed_tokens)
        keywords = set()

        if self.use_pos_tagging:
            if not self.allowed_pos_tags:
                logger.error("POS tagging requested but no allowed tags set. Cannot extract POS-based keywords.")
            else:
                tagged_tokens = self._get_pos_tags(processed_tokens_tuple)
                if tagged_tokens:
                    pos_filtered_tokens = [word for word, tag in tagged_tokens if tag in self.allowed_pos_tags]
                    if pos_filtered_tokens:
                        logger.debug(
                            f"Extracted {len(pos_filtered_tokens)} potential keywords using POS tags: {self.allowed_pos_tags}",
                        )
                        keywords = set(self._normalize_tokens(tuple(pos_filtered_tokens)))
                    else:
                        logger.warning("No tokens matched the allowed POS tags after tagging.")
                else:
                    logger.error("Keyword extraction failed: POS tagging returned no results.")
        else:
            keywords = set(self._normalize_tokens(processed_tokens_tuple))
            if not keywords:
                logger.warning("Normalization returned no tokens.")
            else:
                logger.debug(f"Extracted {len(keywords)} keywords (all non-stopword tokens).")

        return keywords

    def find_matches_and_score(self, paragraph_a: str, paragraph_b: str) -> Dict[str, Union[List[str], float, int]]:
        """Find keywords and calculate keyword coverage and vocabulary cosine scores."""
        logger.info("Attempting to find matches and score from Paragraph A in Paragraph B.")
        logger.debug(f"Paragraph A (start): '{paragraph_a[:60]}...'")
        logger.debug(f"Paragraph B (start): '{paragraph_b[:60]}...'")

        default_result = self._initialize_default_result()
        vocab_cosine_score = self._calculate_vocab_cosine(paragraph_a, paragraph_b)
        default_result["vocabulary_cosine_similarity"] = vocab_cosine_score

        keywords_a_set = self._extract_keywords_from_a(paragraph_a)
        default_result["keywords_from_a_count"] = len(keywords_a_set)

        if not keywords_a_set:
            logger.warning("Could not extract any keywords from Paragraph A for coverage score. Coverage is 0.")
            return default_result

        coverage_result = self._calculate_keyword_coverage(keywords_a_set, paragraph_b)
        default_result.update(coverage_result)

        return default_result

    def _initialize_default_result(self) -> Dict[str, Union[List[str], float, int]]:
        """Initialize the default result dictionary."""
        return {
            "matched_keywords": [],
            "keywords_from_a_count": 0,
            "matched_keyword_count": 0,
            "keyword_coverage_score": 0.0,
            "vocabulary_cosine_similarity": 0.0,
        }

    def _calculate_vocab_cosine(self, paragraph_a: str, paragraph_b: str) -> float:
        """Calculate vocabulary cosine similarity."""
        processed_tokens_a_list = self._preprocess_text(paragraph_a)
        processed_tokens_b_list = self._preprocess_text(paragraph_b)

        if not processed_tokens_a_list and not processed_tokens_b_list:
            logger.warning("Both paragraphs resulted in empty tokens after preprocessing. Vocab cosine score is 0.")
            return 0.0

        set_a, set_b = set(processed_tokens_a_list), set(processed_tokens_b_list)
        r_vector = set_a.union(set_b)
        if not r_vector:
            logger.debug("No common vocabulary (r_vector empty) after preprocessing for vocab cosine.")
            return 0.0

        l1, l2 = [1 if word in set_a else 0 for word in r_vector], [1 if word in set_b else 0 for word in r_vector]
        dot_product = sum(l1[i] * l2[i] for i in range(len(r_vector)))
        denominator = (sum(l1) * sum(l2)) ** 0.5

        return dot_product / denominator if denominator > 0 else 0.0

    def _calculate_keyword_coverage(
        self,
        keywords_a_set: Set[str],
        paragraph_b: str,
    ) -> Dict[str, Union[List[str], float, int]]:
        """Calculate keyword coverage score."""
        processed_tokens_b_list = self._preprocess_text(paragraph_b)
        if not processed_tokens_b_list:
            logger.warning("Paragraph B preprocessing resulted in no tokens. Coverage score is 0.")
            return {"matched_keywords": [], "matched_keyword_count": 0, "keyword_coverage_score": 0.0}

        normalized_tokens_b_set = set(self._normalize_tokens(tuple(processed_tokens_b_list)))
        matched_keywords_set = keywords_a_set.intersection(normalized_tokens_b_set)
        num_matches = len(matched_keywords_set)
        coverage_score = num_matches / len(keywords_a_set) if keywords_a_set else 0.0

        return {
            "matched_keywords": sorted(matched_keywords_set),
            "matched_keyword_count": num_matches,
            "keyword_coverage_score": coverage_score,
        }


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configure Rich Logging ---
    logging.root.handlers.clear()
    LOG_LEVEL = logging.INFO  # Change to DEBUG for more verbose output
    logging.root.setLevel(LOG_LEVEL)

    if _rich_available:
        rich_handler = RichHandler(level=LOG_LEVEL, show_path=False, rich_tracebacks=True, markup=True)
        logging.root.addHandler(rich_handler)
        try:
            from rich.console import Console

            console = Console()
            separator = lambda: console.print("-" * 60, style="dim")
        except ImportError:
            separator = lambda: print("-" * 60)
        logger.info("Keyword Matching Example [bold green](using Rich logging)[/bold green]")
    else:
        logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        separator = lambda: print("-" * 60)
        logger.info("Keyword Matching Example (standard logging - install 'rich' for better output)")

    # --- Example Paragraphs ---
    para_a = """
    Natural Language Processing (NLP) is a fascinating subfield of artificial intelligence.
    Key techniques include tokenization, lemmatization, and part-of-speech tagging.
    These methods help computers understand human language. We love NLP.
    """

    para_b = """
    Understanding language with computers often involves NLP methods. For example,
    lemmatization reduces words to their base form. Artificial intelligence
    is advancing rapidly, especially in language analysis. The dog barked.
    """

    para_c = """
    This paragraph talks about completely different topics, like astrophysics
    and the study of distant galaxies. There should be minimal overlap.
    """

    para_empty = ""

    # --- Helper Function to Print Results ---
    def print_match_results(scenario_name: str, results: Dict) -> None:
        """Print the results of keyword matching for a given scenario.

        Args:
            scenario_name (str): The name of the scenario being evaluated.
            results (Dict): A dictionary containing the results of the keyword matching,
                            including scores and matched keywords.

        """
        logger.info(f"[bold cyan]>>> {scenario_name} Results:[/bold cyan]")
        logger.info(f"  Total Keywords from A (for coverage): {results['keywords_from_a_count']}")
        logger.info(f"  Matched Keywords Count: {results['matched_keyword_count']}")

        # Format coverage score nicely
        cov_score = results["keyword_coverage_score"]
        cov_color = "green" if cov_score > GOOD_KEYWORD_COVERAGE else "yellow" if cov_score > 0 else "red"
        logger.info(f"  Keyword Coverage Score: [{cov_color}]{cov_score:.4f}[/{cov_color}]")

        # Format cosine score nicely
        cos_score = results["vocabulary_cosine_similarity"]
        cos_color = "green" if cos_score > GOOD_VOCAB_COSINE else "yellow" if cos_score > BAD_VOCAB_COSINE else "red"
        logger.info(f"  Vocabulary Cosine Score: [{cos_color}]{cos_score:.4f}[/{cos_color}]")

        if results["matched_keywords"]:
            logger.info(f"  Matched Keywords (for coverage): {results['matched_keywords']}")
        else:
            logger.info("  Matched Keywords (for coverage): None")
        separator()

    # --- Run Matching Scenarios ---

    # Scenario 1: Default settings
    matcher_default = KeywordMatcher()
    results1 = matcher_default.find_matches_and_score(para_a, para_b)
    print_match_results("Scenario 1: Default Settings", results1)

    # Scenario 2: Using POS Tagging
    matcher_pos = KeywordMatcher(use_pos_tagging=True)
    results2 = matcher_pos.find_matches_and_score(para_a, para_b)
    print_match_results("Scenario 2: Using POS Tagging (Nouns & Adjectives)", results2)

    # Scenario 3: No Lemmatization, No POS Tagging
    matcher_simple = KeywordMatcher(use_lemmatization=False, use_pos_tagging=False)
    results3 = matcher_simple.find_matches_and_score(para_a, para_b)
    print_match_results("Scenario 3: No Lemmatization or POS", results3)

    # Scenario 4: Matching against a dissimilar paragraph
    results4 = matcher_default.find_matches_and_score(para_a, para_c)
    print_match_results("Scenario 4: Matching Dissimilar Paragraphs", results4)

    # Scenario 5: Matching with empty input
    results5a = matcher_default.find_matches_and_score(para_a, para_empty)
    print_match_results("Scenario 5a: Matching A vs Empty", results5a)

    results5b = matcher_default.find_matches_and_score(para_empty, para_b)
    print_match_results("Scenario 5b: Matching Empty vs B", results5b)

    logger.info("Keyword Matching Example Finished.")
