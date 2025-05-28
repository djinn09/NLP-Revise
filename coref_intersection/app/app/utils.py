"""Utility functions for natural language processing tasks.

This module provides:
- Functions to extract span words from a document.
- Functions to resolve neural coreferences in a document.
- Helper functions for working with clusters and spans.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
    """Get the indices of the spans in the cluster that contain a noun or proper noun.

    Args:
        doc (Doc): The SpaCy document containing the cluster.
        cluster (List[List[int]]): The cluster of spans.

    Returns:
        List[int]: The indices of the spans in the cluster that contain a noun or proper noun.

    """
    spans = [doc[span[0] : span[1] + 1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    return [i for i, span_pos in enumerate(spans_pos) if any(pos in span_pos for pos in ["NOUN", "PROPN"])]


def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]) -> Tuple[Span, List[int]]:
    """Get the head span and its indices from a cluster of spans.

    Args:
        doc: The spaCy document.
        cluster: The cluster of spans.
        noun_indices: The indices of the noun phrases in the cluster.

    Returns:
        A tuple containing the head span and its indices.

    """
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start : head_end + 1]
    return head_span, [head_start, head_end]
    from spacy.tokens import Doc, Span


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
    noun_indices = get_span_noun_indices(doc, cluster)
    return noun_indices[0] if noun_indices else 0