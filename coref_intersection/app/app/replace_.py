from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple
from . import utils

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


def core_logic_part(
    document: Doc,
    coref: List[int],
    resolved: List[str],
    mention_span: Span,
) -> List[str]:
    """Replace the given coreference span with the mention span in the resolved list of strings.

    Args:
        document (Doc): The SpaCy document containing the coreference span.
        coref (List[int]): The coreference span to replace.
        resolved (List[str]): The list of strings containing the resolved text.
        mention_span (Span): The mention span to replace the coreference span with.

    Returns:
        List[str]: The modified resolved list of strings.

    """
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]) -> bool:
    """Check if the given span contains any other span in the list of all spans.

    Args:
        span (List[int]): The span to check.
        all_spans (List[List[int]]): The list of all spans.

    Returns:
        bool: Whether the span contains any other span or not.

    """
    return any(s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans)


def improved_replace_co_refs(document: Doc, clusters: List[List[List[int]]]) -> str:
    """Resolve coreferences in the document using the provided clusters.

    Args:
        document (Doc): The SpaCy document to resolve coreferences in.
        clusters (List[List[List[int]]]): A list of clusters, where each cluster is a list of spans,
                                          and each span is defined by a list of two integers 
                                          indicating the start and end token indices.

    Returns:
        str: The text of the document with coreferences resolved.

    """
    # Initialize resolved text with the original document text, preserving whitespace
    resolved: List[str] = [tok.text_with_ws for tok in document]

    # Flatten all spans from all clusters into a single list
    all_spans: List[List[int]] = [span for cluster in clusters for span in cluster]

    # Iterate over each cluster to resolve coreferences
    for cluster in clusters:
        # Get indices of spans containing nouns or proper nouns
        noun_indices: List[int] = utils.get_span_noun_indices(document, cluster)

        # If there are noun indices in the cluster, process the cluster
        if noun_indices:
            # Determine the head span and its indices
            mention_span, mention = utils.get_cluster_head(document, cluster, noun_indices)

            # Iterate over each coreference in the cluster
            for coref in cluster:
                # Replace coreference with mention span if it's not the mention itself
                # and does not contain other spans
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    # Join the resolved text list into a single string and return
    return "".join(resolved)


def original_replace_corefs(document: Doc, clusters: List[List[List[int]]]) -> str:
    """Resolve coreferences in the document using the provided clusters.

    Args:
        document (Doc): The SpaCy document to resolve coreferences in.
        clusters (List[List[List[int]]]): A list of clusters, where each cluster is a list of spans,
                                          and each span is defined by a list of two integers
                                          indicating the start and end token indices.

    Returns:
        str: The text of the document with coreferences resolved.

    """
    # Initialize resolved text with the original document text, preserving whitespace
    resolved: List[str] = [tok.text_with_ws for tok in document]

    # Iterate over each cluster to resolve coreferences
    for cluster in clusters:
        # The first span in the cluster is the mention span
        mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
        mention_span = document[mention_start:mention_end]

        # Iterate over each coreference in the cluster
        for coref in cluster[1:]:
            # Replace coreference with mention span
            core_logic_part(document, coref, resolved, mention_span)

    # Join the resolved text list into a single string and return
    return "".join(resolved)
