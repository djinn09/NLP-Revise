from app.intersection import IntersectionStrategy


def get_span_words(span, document):
    return " ".join(document[span[0] : span[1] + 1])


def get_neural_reference_resolved(doc):

    neural_response = {}
    # mentions = [
    #     {
    #         "start": mention.start_char,
    #         "end": mention.end_char,
    #         "text": mention.text,
    #         "resolved": cluster.main.text,
    #     }
    #     for cluster in doc._.coref_clusters
    #     for mention in cluster.mentions
    # ]
    clusters = list(
        (cluster.main.text, list(span.text for span in cluster))
        for cluster in doc._.coref_clusters
    )
    resolved = doc._.coref_resolved
    # neural_response["mentions"] = mentions
    neural_response["clusters"] = clusters
    neural_response["resolved"] = resolved
    return neural_response


def get_cluster_head_idx(doc, cluster):
    noun_indices = IntersectionStrategy.get_span_noun_indices(doc, cluster)
    return noun_indices[0] if noun_indices else 0


def print_clusters(doc, clusters):
    def get_span_words(span, allen_document):
        return " ".join(allen_document[span[0] : span[1] + 1])

    allen_document, clusters = [t.text for t in doc], clusters
    for cluster in clusters:
        cluster_head_idx = get_cluster_head_idx(doc, cluster)
        if cluster_head_idx >= 0:
            cluster_head = cluster[cluster_head_idx]
            print(get_span_words(cluster_head, allen_document) + " - ", end="")
            print("[", end="")
            for i, span in enumerate(cluster):
                print(
                    get_span_words(span, allen_document)
                    + ("; " if i + 1 < len(cluster) else ""),
                    end="",
                )
            print("]")
