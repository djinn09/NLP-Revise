from typing import List

from spacy.tokens import Doc, Span


def core_logic_part(
    document: Doc, coref: List[int], resolved: List[str], mention_span: Span
):
    final_token = document[coref[1]]
    if final_token.tag_ in ["PRP$", "POS"]:
        resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
    else:
        resolved[coref[0]] = mention_span.text + final_token.whitespace_
    for i in range(coref[0] + 1, coref[1] + 1):
        resolved[i] = ""
    return resolved


def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
    spans = [doc[span[0] : span[1] + 1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [
        i
        for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ["NOUN", "PROPN"])
    ]
    return span_noun_indices


def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
    return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])


def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
    head_idx = noun_indices[0]
    head_start, head_end = cluster[head_idx]
    head_span = doc[head_start : head_end + 1]
    return head_span, [head_start, head_end]


def improved_replace_corefs(document, clusters):
    """
    Nested coreferent mentions
    """
    resolved = list(tok.text_with_ws for tok in document)
    all_spans = [
        span for cluster in clusters for span in cluster
    ]  # flattened list of all spans

    for cluster in clusters:
        noun_indices = get_span_noun_indices(document, cluster)

        if noun_indices:
            mention_span, mention = get_cluster_head(document, cluster, noun_indices)

            for coref in cluster:
                if coref != mention and not is_containing_other_spans(coref, all_spans):
                    core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


def original_replace_corefs(document: Doc, clusters: List[List[List[int]]]) -> str:
    resolved = list(tok.text_with_ws for tok in document)

    for cluster in clusters:
        mention_start, mention_end = cluster[0][0], cluster[0][1] + 1
        mention_span = document[mention_start:mention_end]

        for coref in cluster[1:]:
            core_logic_part(document, coref, resolved, mention_span)

    return "".join(resolved)


