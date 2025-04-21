"""Utility functions for natural language processing tasks.

This module provides:
- Functions to extract span words from a document.
- Functions to resolve neural coreferences in a document.
- Helper functions for working with clusters and spans.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List

from app.intersection import IntersectionStrategy

if TYPE_CHECKING:
    from spacy.tokens import Doc


def get_span_words(span: List[int], document: List[str]) -> str:
    """Return a string consisting of the words in the given span.

    Args:
        span (List[int]): A list of two integers representing the start and end of the span.
        document (List[str]): A list of the words in the document.

    Returns:
        str: A string consisting of the words in the given span.

    """
    return " ".join(document[span[0] : span[1] + 1])


def get_neural_reference_resolved(doc: Doc) -> dict:
    """Resolve neural coreferences in the document.

    Args:
        doc (Doc): The SpaCy document containing coreference clusters.

    Returns:
        dict: A dictionary containing the resolved coreferences, including clusters and resolved text.

    """
    neural_response: dict = {}
    clusters = [(cluster.main.text, [span.text for span in cluster]) for cluster in doc._.coref_clusters]
    resolved: str = doc._.coref_resolved
    neural_response["clusters"] = clusters
    neural_response["resolved"] = resolved
    return neural_response


def get_cluster_head_idx(doc: Doc, cluster: List[List[int]]) -> int:
    """Get the index of the head span in a cluster of spans.

    The head span is defined as the first noun phrase in the cluster.

    Args:
        doc (Doc): The spaCy document containing the text.
        cluster (List[List[int]]): The cluster of spans from which to extract the head.

    Returns:
        int: The index of the head span in the cluster.

    """
    noun_indices = IntersectionStrategy.get_span_noun_indices(doc, cluster)
    return noun_indices[0] if noun_indices else 0