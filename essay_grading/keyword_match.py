from __future__ import annotations

import difflib
import logging
import os
import re
import string
from collections import Counter
from itertools import zip_longest
from typing import Callable

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
    """A class to compute TF-IDF-based similarity between two texts.

    Attributes
    ----------
    original : str
        The original text to compare.
    compare_text : str
        The text to compare against the original.
    lemmatization : bool, optional
        Whether to use lemmatization for text normalization (default is False).
    without_normalized : bool, optional
        Whether to skip normalization during vectorization (default is False).

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
    ) -> None:
        """Initialize a TFIDF instance.

        Parameters
        ----------
        original : str
            The original text to compare.
        compare_text : str
            The text to compare against the original.
        lemmatization : bool, optional
            Whether to use lemmatization for text normalization (default is False).
        without_normalized : bool, optional
            Whether to skip normalization during vectorization (default is False).

        """
        self.normalize = self.lemma_normalize if lemmatization else self.stem_normalize
        self.original = original
        self.compare_text = compare_text
        self.without_normalized = without_normalized

    def lemmatize_tokens(self, tokens: list[str], lemmatizer: Callable = word_net_lemmatizer) -> list[str]:
        """Apply lemmatization to the given list of tokens.

        Parameters
        ----------
        tokens : list[str]
            The list of tokens to lemmatize.
        lemmatizer : callable, optional
            The lemmatizer to use (default is `nltk.stem.WordNetLemmatizer()`).

        Returns
        -------
        lemmatized_tokens : list[str]
            The lemmatized tokens.

        """
        return [lemmatizer.lemmatize(tok_en) for tok_en in tokens]  # Apply lemmatization

    @staticmethod
    def stem_normalize(text: str) -> list[str]:
        """Normalize text by stemming tokens and removing punctuation.

        Parameters
        ----------
        text : str
            The input text to normalize.

        Returns
        -------
        list[str]
            A list of stemmed tokens from the input text.

        """
        tokens = nltk.word_tokenize(text.lower().translate(remove_punctuation_map))
        return [stemmer.stem(item) for item in tokens]

    @staticmethod
    def lemma_normalize(text: str) -> list[str]:
        """Normalize text by lemmatizing tokens and removing punctuation.

        Parameters
        ----------
        text : str
            The input text to normalize.

        Returns
        -------
        list[str]
            A list of lemmatized tokens from the input text.

        """
        tokens = nltk.word_tokenize(text.lower().translate(remove_punctuation_map))

        return [lemmatizer.lemmatize(tok_en) for tok_en in tokens]

    def vectorizer(self, *, without_normalized: bool = True) -> TfidfVectorizer:
        """Create a TF-IDF vectorizer based on the normalization setting.

        - without_normalized=True: use regex tokenization only.
        - without_normalized=False: use the custom normalize() tokenizer.
        """
        if without_normalized:
            token_pattern = os.getenv("TOKEN_PATTERN", r"\w+")
            # Only token_pattern, no tokenizer
            return TfidfVectorizer(token_pattern=token_pattern, tokenizer=None)
        # Only tokenizer, no token_pattern
        return TfidfVectorizer(tokenizer=self.normalize, token_pattern=None)

    def fit(self):
        """Fit the TF-IDF vectorizer on the original and comparison texts.

        Returns
        -------
        tfidf : scipy.sparse.csr.csr_matrix
            The TF-IDF matrix of the original and comparison texts.

        """
        vectorizer: TfidfVectorizer = self.vectorizer(without_normalized=self.without_normalized)
        return vectorizer.fit_transform([self.original, self.compare_text])

    def calculate_distance(self, metric: str) -> float:
        """Calculate the similarity or distance between the original and comparison texts.

        Parameters
        ----------
        metric : str
            The metric to use for calculating distance. Supported values are "cosine",
            "euclidean", "manhattan", "minkowski", "jaccard", and "hamming".

        Returns
        -------
        float
            The calculated similarity or distance value.

        Raises
        ------
        ValueError
            If an invalid metric is provided.

        """
        tfidf = self.fit()
        if metric == "cosine":
            print("Calculating TFIDF cosine similarity...")  # noqa: T201
            return cosine_similarity(tfidf[0], tfidf[1])[0][0]
        if metric in ("euclidean", "manhattan", "minkowski"):
            print(f"Calculating TFIDF {metric} distance...")
            return pairwise_distances(tfidf.toarray(), metric=metric)[0, 1]
        if metric == "jaccard":
            print("Calculating TFIDF jaccard distance...")
            tokens1 = set(self.normalize(self.original))
            tokens2 = set(self.normalize(self.compare_text))
            common_tokens = list(tokens1.intersection(tokens2))
            score = jaccard_score(common_tokens, common_tokens, average="micro")
            return float(score) if isinstance(score, (int, float)) else score.item()
        if metric == "hamming":
            print("Calculating TFIDF hamming distance...")
            # Get the longer sentence length
            sentence1 = self.original
            sentence2 = self.compare_text

            max_length = max(len(sentence1), len(sentence2))

            # Create empty dictionaries to store character counts
            char_counts1 = Counter(list(sentence1))
            char_counts2 = Counter(list(sentence2))

            # Calculate the Hamming distance
            distance = 0
            for k1, k2 in zip_longest(char_counts1.keys(), char_counts2.keys()):
                # Consider difference in character occurrences and handle missing characters
                distance += abs(char_counts1.get(k1, 0) - char_counts2.get(k2, 0))
            return distance / max_length
        msg = "Invalid metric"
        raise ValueError(msg)


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
