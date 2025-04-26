from __future__ import annotations

import difflib
import logging
import os
import re
import string
from collections import Counter
from itertools import zip_longest
from typing import Callable, List, Optional, Tuple
from regex import P
from scipy.sparse import csr_matrix
import nltk
import rapidfuzz
from fuzzywuzzy import fuzz as fuzzy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rank_bm25 import BM25L as BM25
from rapidfuzz import fuzz
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

stop_words = set(stopwords.words("english"))


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


class BleuScore:
    """A class used to score sentences based on the input keyword or between two strings."""

    def pre_process_text(self, text: str) -> list[str]:
        """Preprocesses the input text by removing special characters, tokenizing words, and lemmatizing them.

        Args:
            text (str): The input text to preprocess.

        Returns:
            list[str]: A list of lemmatized words from the input text.

        """
        text = re.sub(SPECIAL_CHARS_REMOVE_PATTERN, "", text)
        try:
            # Tokenize words in a sentence
            word_tokens = word_tokenize(text)
            # Lemmatization of words
            return [
                word_net_lemmatizer.lemmatize(re.sub(SPECIAL_CHARS_REMOVE_PATTERN, "", w))
                for w in word_tokens
                if w not in stop_words
            ]
        except (ValueError, TypeError):  # type: ignore  # noqa: PGH003
            logging.exception("Error occurred in text preprocessing")  # noqa: LOG015
            return []

    def score_text(self, text1: str, text2: str) -> float:
        """Compare two sentences and return their BLEU score.

        Parameters
        ----------
        text1 : str
            The first sentence to compare.
        text2 : str
            The second sentence to compare.

        Returns
        -------
        float
            The BLEU score of the two sentences.

        Notes
        -----
        BLEU score is a measure of how similar two sentences are. It is calculated by comparing
        the n-grams of the two sentences. The BLEU score is a value between 0 and 1, where 1 is
        a perfect match and 0 indicates no similarity at all.

        """
        try:
            # Tokenization and Lemmatization of text1 and text2
            word_list1 = self.pre_process_text(text1)
            wordlist2 = self.pre_process_text(text2)

            reference = [word_list1]  # Reference is text1
            chencherry = SmoothingFunction()

            # Calculate BLEU score
            return sentence_bleu(reference, wordlist2, smoothing_function=chencherry.method1)

        except (ValueError, TypeError, NltkError):
            logging.exception("Error occurred in text preprocessing")  # noqa: LOG015
            return 0.0

    # similarity of subject
    def score_text_ngram(self, text1: str, text2: str) -> float:
        """Calculate the BLEU score between two sentences with n-grams.

        Parameters
        ----------
        text1 : str
            The first sentence to compare.
        text2 : str
            The second sentence to compare.

        Returns
        -------
        float
            The BLEU score of the two sentences with n-grams between 1 and 4.

        Notes
        -----
        BLEU score is a measure of how similar two sentences are. It is calculated by comparing
        the n-grams of the two sentences. The BLEU score is a value between 0 and 1, where 1 is
        a perfect match and 0 indicates no similarity at all.

        """
        try:
            # Tokenization and Lemmatization of the keyword
            keyword_list = self.pre_process_text(text1)

            # Tokenization and Lemmatization of the sentences
            wordlist = self.pre_process_text(text2)
            reference = [keyword_list]
            chencherry = SmoothingFunction()
            # sentence bleu calculates the score based on 1-gram,2-gram,3-gram-4-gram,
            # and a cumulative of the above is taken as score of the sentence.
            bleu_score_1 = sentence_bleu(
                reference,
                wordlist,
                weights=(1, 0, 0, 0),
                smoothing_function=chencherry.method1,
            )
            bleu_score_2 = sentence_bleu(
                reference,
                wordlist,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=chencherry.method1,
            )
            bleu_score_3 = sentence_bleu(
                reference,
                wordlist,
                weights=(0.33, 0.33, 0.34, 0),
                smoothing_function=chencherry.method1,
            )
            bleu_score_4 = sentence_bleu(
                reference,
                wordlist,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=chencherry.method1,
            )
            return (
                4 * float(bleu_score_4) + 3 * float(bleu_score_3) + 2 * float(bleu_score_2) + float(bleu_score_1)
            ) / 10

        except (ValueError, TypeError):
            logging.exception("Error occurred in score_text_n_gram")  # noqa: LOG015
            return 0.0


def extract_string_similarity_vector(original: str, compare_text: str) -> dict[str, float]:
    """Extract various string similarity metrics between two texts.

    Parameters
    ----------
    original : str
        The original text to compare.
    compare_text : str
        The text to compare against the original.

    Returns
    -------
    dict[str, float]
        A dictionary containing similarity metrics and their respective scores.

    """
    print("Extracting string similarity vector...")  # noqa: T201
    # Initialize the similarity metrics

    s1 = original.lower()
    s2 = compare_text.lower()
    print(f"Original: {s1}")  # noqa: T201
    print(f"Compare: {s2}")  # noqa: T201
    # Initialize the similarity metrics
    seq = difflib.SequenceMatcher(None, s1, s2)

    bm25 = BM25([s1.split()])
    tokenized_query = s2.split()
    doc_scores = bm25.get_scores(tokenized_query)
    print(f"BM25 scores: {doc_scores}")  # noqa: T201
    # Initialize the TF-IDF vectorizer
    tfidf = TFIDF(s1, s2)
    scorer = BleuScore()
    return {
        "levenshtein": normalized_levenshtein.similarity(s1, s2),
        "jaro_winkler": jaro_winkler.similarity(s1, s2),
        "jaro_similarity": rapidfuzz.distance.JaroWinkler.distance(s1, s2),
        "metric_lcs": metric_lcs.distance(s1, s2),
        "qgram2": qgram2.distance(s1, s2),
        "qgram3": qgram3.distance(s1, s2),
        "qgram4": qgram4.distance(s1, s2),
        "jaccard": jaccard.similarity(s1, s2),
        "cosine": sim_cosine.similarity(s1, s2),
        "partial_ratio": rapidfuzz.fuzz.partial_ratio(s1, s2),
        "partial_token_set_ratio": rapidfuzz.fuzz.partial_token_set_ratio(s1, s2),
        "partial_token_sort_ratio": rapidfuzz.fuzz.partial_token_sort_ratio(s1, s2),
        "token_set_ratio": rapidfuzz.fuzz.token_set_ratio(s1, s2),
        "token_sort_ratio": rapidfuzz.fuzz.token_sort_ratio(s1, s2),
        "QRatio": fuzz.QRatio(s1, s2),
        "UQRatio": fuzzy.UQRatio(s1, s2),
        "UWRatio": fuzzy.UWRatio(s1, s2),
        "fuzzwuzzy": fuzz.ratio(s1, s2),
        "WRatio": fuzz.WRatio(s1, s2),
        "seq_match": seq.ratio(),
        "bleu_score": scorer.score_text_ngram(s1, s2),
        "bm25": doc_scores[0],
        "Cosine": tfidf.calculate_distance("cosine"),
        "Euclidean": tfidf.calculate_distance("euclidean"),
        "Manhattan": tfidf.calculate_distance("manhattan"),
        "Minkowski": tfidf.calculate_distance("minkowski"),
        "Jaccard": tfidf.calculate_distance("jaccard"),
        "Hamming": tfidf.calculate_distance("hamming"),
    }


if __name__ == "__main__":
    # Example usage
    original_text = "This is a sample text."
    compare_text = "This is a sample text for comparison."
    similarity_vector = extract_string_similarity_vector(original_text, compare_text)
    print(similarity_vector)  # noqa: T201
    # Output: A dictionary containing various similarity metrics and their scores
    # Note: The output will vary based on the input texts and the similarity metrics used.
    # You can adjust the original_text and compare_text variables to test different cases.
