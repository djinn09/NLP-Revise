from allennlp.predictors.predictor import Predictor
from app.intersection import (
    FuzzyIntersectionStrategy,
    PartialIntersectionStrategy,
    StrictIntersectionStrategy,
    IntersectionStrategy,
)
from app.utils import get_span_words
from app.replace_ import improved_replace_corefs


def get_coref_object(path):
    predictor = Predictor.from_path(path)
    return predictor


def get_coref_prediction(predictor, text):
    prediction = predictor.predict(document=text)
    document, clusters = prediction["document"], prediction["clusters"]
    clusters = [
        (
            get_span_words(cluster[0], document),
            [get_span_words(span, document) for span in cluster],
        )
        for cluster in clusters
        if len(cluster) > 1
    ]
    coref_res = predictor.coref_resolved(text)

    return clusters, coref_res, prediction["clusters"]


def get_coref_intersection(predictor, nlp, strategy, text):
    if strategy == "strict":
        strategy = StrictIntersectionStrategy(predictor, nlp)
    elif strategy == "partial":
        strategy = PartialIntersectionStrategy(predictor, nlp)
    elif strategy == "fuzzy":
        strategy = FuzzyIntersectionStrategy(predictor, nlp)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return strategy.clusters(text)


def get_improved_coref(doc, clusters):
    return improved_replace_corefs(doc, clusters)


def get_cluster_head_idx(doc, cluster):
    noun_indices = IntersectionStrategy.get_span_noun_indices(doc, cluster)
    return noun_indices[0] if noun_indices else 0


def get_clusters(doc, clusters):
    def get_span_words(span, allen_document):
        return " ".join(allen_document[span[0] : span[1] + 1])

    allen_document, clusters = [t.text for t in doc], clusters
    new_clusters = []
    for cluster in clusters:
        cluster_head_idx = get_cluster_head_idx(doc, cluster)
        if cluster_head_idx >= 0:
            cluster_head = cluster[cluster_head_idx]
            key = get_span_words(cluster_head, allen_document)
            value = []
            for i, span in enumerate(cluster):
                value.append(get_span_words(span, allen_document))
            new_clusters.append((key, value))
    return new_clusters


def get_allennlp_coref(predictor, nlp, text):
    clusters, coref_res, int_clusters = get_coref_prediction(predictor, text)
    doc = nlp(text)

    return {
        "clusters": clusters,
        "coref_res": coref_res,
        "improved_coref_res": get_improved_coref(doc, int_clusters),
        "strict": get_clusters(
            doc, get_coref_intersection(predictor, nlp, "strict", text)
        ),
        "partial": get_clusters(
            doc, get_coref_intersection(predictor, nlp, "partial", text)
        ),
        "fuzzy": get_clusters(
            doc, get_coref_intersection(predictor, nlp, "fuzzy", text)
        ),
    }


if __name__ == "__main__":
    from main import nlp

    predictor = get_coref_object("models/coref-spanbert-large-2021.03.10.tar.gz")
    text = """Every Tuesday and Friday, Recode’s Kara Swisher and NYU Professor Scott Galloway offer sharp, unfiltered insights into the biggest stories in tech, business, and politics. They make bold predictions, pick winners and losers, and bicker and banter like no one else. Kara is out welcoming the newest member of the Pivot family! Scott is joined by co-host Stephanie Ruhle to talk about The Great Resignation, inflation, J&J’s split, and Steve Bannon’s indictment. Also, Elon is still bullying senators on Twitter, and Beto is officially running for Governor of Texas. Plus, Scott chats with Friend of Pivot, Founder and CEO of Boom Supersonic, Blake Scholl about supersonic air travel."""
    # clusters, coref_res, int_clusters = get_coref_prediction(predictor, text)
    # strict = StrictIntersectionStrategy(predictor, nlp)
    # partial = PartialIntersectionStrategy(predictor, nlp)
    # fuzzy = FuzzyIntersectionStrategy(predictor, nlp)
    # doc = nlp(text)
    # for i in ["strict", "partial", "fuzzy"]:
    #     print(f"{i} strategy:")
    #     print_clusters(doc, get_coref_intersection(predictor, nlp, i, text))
    # print(clusters)
    # print(coref_res)
    # print(get_improved_coref(doc, int_clusters))
    print(get_allennlp_coref(predictor, nlp, text))
