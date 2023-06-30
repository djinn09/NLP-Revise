from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List

from spacy.tokens import Doc


class IntersectionStrategy(ABC):
    def __init__(self, allen_model, hugging_model):
        self.allen_clusters = []
        self.hugging_clusters = []
        self.allen_model = allen_model
        self.hugging_model = hugging_model
        self.document = []
        self.doc = None

    @abstractmethod
    def get_intersected_clusters(self):
        raise NotImplementedError

    @staticmethod
    def get_span_noun_indices(doc: Doc, cluster: List[List[int]]):
        spans = [doc[span[0] : span[1] + 1] for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [
            i
            for i, span_pos in enumerate(spans_pos)
            if any(pos in span_pos for pos in ["NOUN", "PROPN"])
        ]
        return span_noun_indices

    @staticmethod
    def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]):
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start : head_end + 1]
        return head_span, [head_start, head_end]

    @staticmethod
    def is_containing_other_spans(span: List[int], all_spans: List[List[int]]):
        return any(
            [s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans]
        )

    def coref_resolved_improved(self, doc: Doc, clusters: List[List[List[int]]]):
        resolved = [tok.text_with_ws for tok in doc]
        all_spans = [
            span for cluster in clusters for span in cluster
        ]  # flattened list of all spans

        for cluster in clusters:
            noun_indices = self.get_span_noun_indices(doc, cluster)
            if noun_indices:
                mention_span, mention = self.get_cluster_head(
                    doc, cluster, noun_indices
                )

                for coref in cluster:
                    if coref != mention and not self.is_containing_other_spans(
                        coref, all_spans
                    ):
                        final_token = doc[coref[1]]
                        if final_token.tag_ in ["PRP$", "POS"]:
                            resolved[coref[0]] = (
                                mention_span.text + "'s" + final_token.whitespace_
                            )
                        else:
                            resolved[coref[0]] = (
                                mention_span.text + final_token.whitespace_
                            )

                        for i in range(coref[0] + 1, coref[1] + 1):
                            resolved[i] = ""

        return "".join(resolved)

    def clusters(self, text):
        self.acquire_models_clusters(text)
        return self.get_intersected_clusters()

    def resolve_coreferences(self, text: str):
        clusters = self.clusters(text)
        resolved_text = self.coref_resolved_improved(self.doc, clusters)
        return resolved_text

    def acquire_models_clusters(self, text: str):
        allen_prediction = self.allen_model.predict(text)
        self.allen_clusters = allen_prediction["clusters"]
        self.document = allen_prediction["document"]
        self.doc = self.hugging_model(text)
        hugging_clusters = (
            self._transform_huggingface_answer_to_allen_list_of_clusters()
        )
        self.hugging_clusters = hugging_clusters

    def _transform_huggingface_answer_to_allen_list_of_clusters(self):
        list_of_clusters = []
        for cluster in self.doc._.coref_clusters:
            list_of_clusters.append([])
            for span in cluster:
                list_of_clusters[-1].append([span[0].i, span[-1].i])
        return list_of_clusters


class PartialIntersectionStrategy(IntersectionStrategy):
    def get_intersected_clusters(self):
        intersected_clusters = []
        for allen_cluster in self.allen_clusters:
            intersected_cluster = []
            for hugging_cluster in self.hugging_clusters:
                allen_set = set(tuple([tuple(span) for span in allen_cluster]))
                hugging_set = set(tuple([tuple(span) for span in hugging_cluster]))
                intersect = sorted(
                    [list(el) for el in allen_set.intersection(hugging_set)]
                )
                if len(intersect) > 1:
                    intersected_cluster += intersect
            if intersected_cluster:
                intersected_clusters.append(intersected_cluster)
        return intersected_clusters


class FuzzyIntersectionStrategy(PartialIntersectionStrategy):
    """Is treated as a PartialIntersectionStrategy, yet first must map AllenNLP spans and Huggingface spans."""

    @staticmethod
    def flatten_cluster(list_of_clusters):
        return [span for cluster in list_of_clusters for span in cluster]

    def _check_whether_spans_are_within_range(self, allen_span, hugging_span):
        allen_range = range(allen_span[0], allen_span[1] + 1)
        hugging_range = range(hugging_span[0], hugging_span[1] + 1)
        allen_within = allen_span[0] in hugging_range and allen_span[1] in hugging_range
        hugging_within = (
            hugging_span[0] in allen_range and hugging_span[1] in allen_range
        )
        return allen_within or hugging_within

    def _add_span_to_list_dict(self, allen_span, hugging_span):
        if allen_span[1] - allen_span[0] > hugging_span[1] - hugging_span[0]:
            self._add_element(allen_span, hugging_span)
        else:
            self._add_element(hugging_span, allen_span)

    def _add_element(self, key_span, val_span):
        if tuple(key_span) in self.swap_dict_list.keys():
            self.swap_dict_list[tuple(key_span)].append(tuple(val_span))
        else:
            self.swap_dict_list[tuple(key_span)] = [tuple(val_span)]

    def _filter_out_swap_dict(self):
        swap_dict = {}
        for key, vals in self.swap_dict_list.items():
            if self.swap_dict_list[key] != vals[0]:
                swap_dict[key] = sorted(vals, key=lambda x: x[1] - x[0], reverse=True)[
                    0
                ]
        return swap_dict

    def _swap_mapped_spans(self, list_of_clusters, model_dict):
        for cluster_idx, cluster in enumerate(list_of_clusters):
            for span_idx, span in enumerate(cluster):
                if tuple(span) in model_dict.keys():
                    list_of_clusters[cluster_idx][span_idx] = list(
                        model_dict[tuple(span)]
                    )
        return list_of_clusters

    def get_mapped_spans_in_lists_of_clusters(self):
        self.swap_dict_list = {}
        for allen_span in self.flatten_cluster(self.allen_clusters):
            for hugging_span in self.flatten_cluster(self.hugging_clusters):
                if self._check_whether_spans_are_within_range(allen_span, hugging_span):
                    self._add_span_to_list_dict(allen_span, hugging_span)
        swap_dict = self._filter_out_swap_dict()

        allen_clusters_mapped = self._swap_mapped_spans(
            deepcopy(self.allen_clusters), swap_dict
        )
        hugging_clusters_mapped = self._swap_mapped_spans(
            deepcopy(self.hugging_clusters), swap_dict
        )
        return allen_clusters_mapped, hugging_clusters_mapped

    def get_intersected_clusters(self):
        (
            allen_clusters_mapped,
            hugging_clusters_mapped,
        ) = self.get_mapped_spans_in_lists_of_clusters()
        self.allen_clusters = allen_clusters_mapped
        self.hugging_clusters = hugging_clusters_mapped
        return super().get_intersected_clusters()


class StrictIntersectionStrategy(IntersectionStrategy):
    def get_intersected_clusters(self):
        intersected_clusters = []
        for allen_cluster in self.allen_clusters:
            for hugging_cluster in self.hugging_clusters:
                if allen_cluster == hugging_cluster:
                    intersected_clusters.append(allen_cluster)
        return intersected_clusters


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
