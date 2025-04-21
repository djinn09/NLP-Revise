"""Module provides strategies for intersecting coreference resolution clusters from different models (AllenNLP and Huggingface).

It includes:
- IntersectionStrategy: Abstract base class for intersection strategies.
- PartialIntersectionStrategy: Strategy for partial intersection of clusters.
- FuzzyIntersectionStrategy: Strategy for fuzzy intersection with span mapping.
- StrictIntersectionStrategy: Strategy for strict intersection of clusters.
- Utility functions for processing and resolving coreference clusters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from spacy.tokens import Doc, Span


class IntersectionStrategy(ABC):
    """Abstract base class for intersection strategies.

    This class provides a blueprint for implementing different strategies
    to intersect coreference resolution clusters from AllenNLP and Huggingface models.

    Attributes:
        allen_clusters (List[List[List[int]]]): Coreference clusters from the AllenNLP model.
        hugging_clusters (List[List[List[int]]]): Coreference clusters from the Huggingface model.
        allen_model (Predictor): AllenNLP coreference resolution model predictor.
        hugging_model (LanguageModel): Huggingface language model for coreference resolution.
        document (List[str]): Document tokens predicted by the AllenNLP model.
        doc (Doc): SpaCy document containing the text and Huggingface coreference clusters.

    Methods:
        get_intersected_clusters() -> List[List[List[int]]]:
            Abstract method to get intersected clusters between the two models.
        coref_resolved_improved(doc: Doc, clusters: List[List[List[int]]]) -> str:
            Resolve coreferences in the document using improved logic.
        clusters(text: str) -> List[List[List[int]]]:
            Get the intersected coreference clusters from both models.
        resolve_coreferences(text: str) -> str:
            Resolve coreferences in the given text using the intersected clusters.
        acquire_models_clusters(text: str) -> None:
            Acquire the coreference clusters from both models for the given text.

    """

    def __init__(self, allen_model, hugging_model):
        """Initialize an IntersectionStrategy instance.

        Args:
            allen_model (Predictor): AllenNLP coreference resolution model predictor.
            hugging_model (LanguageModel): Huggingface language model for coreference resolution.

        Returns:
            None

        """
        # Initialize empty lists to store AllenNLP and Huggingface coreference clusters
        self.allen_clusters = []
        self.hugging_clusters = []

        # Store the AllenNLP coreference resolution model predictor and Huggingface language model
        self.allen_model = allen_model
        self.hugging_model = hugging_model

        # Initialize empty list to store the document tokens predicted by the AllenNLP model
        self.document = []

        # Initialize None for the SpaCy document containing the text and the Huggingface coreference clusters
        self.doc = None

    @abstractmethod
    def get_intersected_clusters(self: IntersectionStrategy) -> List[List[List[int]]]:
        """Abstract method to get intersected clusters between AllenNLP and Huggingface coreference resolution models.

        Returns:
            List[List[List[int]]]: Intersected clusters between AllenNLP and Huggingface coreference resolution models,
            where each inner list is a cluster containing spans, and each span is a list of two integers representing the
                start and end token indices of the span.

        """
        raise NotImplementedError

    @staticmethod
    def get_span_noun_indices(doc: Doc, cluster: List[List[int]]) -> List[int]:
        """Get the indices of the spans in the cluster that contain a noun or proper noun.

        Args:
            doc (Doc): The SpaCy document containing the cluster.
            cluster (List[List[int]]): The cluster of spans.

        Returns:
            List[int]: The indices of the spans in the cluster that contain a noun or proper noun.

        """
        spans: List[Span] = [doc[span[0] : span[1] + 1] for span in cluster]
        spans_pos: List[List[str]] = [[token.pos_ for token in span] for span in spans]
        span_noun_indices: List[int] = [
            i for i, span_pos in enumerate(spans_pos) if any(pos in span_pos for pos in ["NOUN", "PROPN"])
        ]
        return span_noun_indices

    @staticmethod
    def get_cluster_head(doc: Doc, cluster: List[List[int]], noun_indices: List[int]) -> Tuple[Span, List[int]]:
        """Get the head span and its indices from a cluster of spans.

        Args:
            doc (Doc): The spaCy document containing the text.
            cluster (List[List[int]]): The cluster of spans from which to extract the head.
            noun_indices (List[int]): The indices of noun phrases in the cluster.

        Returns:
            Tuple[Span, List[int]]: A tuple containing the head span and its start and end indices.

        """
        head_idx = noun_indices[0]
        head_start, head_end = cluster[head_idx]
        head_span = doc[head_start : head_end + 1]
        return head_span, [head_start, head_end]

    @staticmethod
    def is_containing_other_spans(span: List[int], all_spans: List[List[int]]) -> bool:
        """Check if the given span contains any other span in the list of all spans.

        Args:
            span (List[int]): The span to check.
            all_spans (List[List[int]]): The list of all spans.

        Returns:
            bool: Whether the span contains any other span or not.

        """
        return any(s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans)

    def coref_resolved_improved(self, doc: Doc, clusters: List[List[List[int]]]) -> str:
        """Resolve coreferences in the document using improved logic.

        Args:
            doc (Doc): The SpaCy document containing the text to process.
            clusters (List[List[List[int]]]): A list of coreference clusters, where each cluster is a list of spans,
                                            and each span is a list of two integers representing the start and end
                                            token indices.

        Returns:
            str: The text of the document with coreferences resolved by replacing coreferring expressions with the
                main mention of each cluster.

        The function processes each cluster to identify the main mention and replaces other coreferences with this
        mention, ensuring possessive forms are handled appropriately. If a coreference is not the main mention and
        does not contain other coreferences, it gets replaced by the main mention.

        """
        resolved = [tok.text_with_ws for tok in doc]
        all_spans = [span for cluster in clusters for span in cluster]  # flattened list of all spans

        for cluster in clusters:
            noun_indices = self.get_span_noun_indices(doc, cluster)
            if noun_indices:
                mention_span, mention = self.get_cluster_head(doc, cluster, noun_indices)

                for coref in cluster:
                    if coref != mention and not self.is_containing_other_spans(coref, all_spans):
                        final_token = doc[coref[1]]
                        if final_token.tag_ in ["PRP$", "POS"]:
                            resolved[coref[0]] = mention_span.text + "'s" + final_token.whitespace_
                        else:
                            resolved[coref[0]] = mention_span.text + final_token.whitespace_

                        for i in range(coref[0] + 1, coref[1] + 1):
                            resolved[i] = ""

        return "".join(resolved)

    def clusters(self, text: str) -> List[List[List[int]]]:
        """Get the intersected coreference clusters from both models.

        Args:
            text (str): The text to process.

        Returns:
            List[List[List[int]]]: A list of coreference clusters, where each cluster is a list of spans,
                                and each span is a list of two integers representing the start and end
                                token indices.

        """
        self.acquire_models_clusters(text)
        return self.get_intersected_clusters()

    def resolve_coreferences(self, text: str) -> str:
        """Resolve coreferences in the given text using the intersected clusters from both models.

        Args:
            text (str): The text to process.

        Returns:
            str: The text with coreferences resolved.

        """
        clusters: List[List[List[int]]] = self.clusters(text)
        resolved_text: str = self.coref_resolved_improved(self.doc, clusters)
        return resolved_text

    def acquire_models_clusters(self, text: str) -> None:
        """Acquire the coreference clusters from both the AllenNLP and Huggingface models for the given text.

        Args:
            text (str): The text to process.

        Attributes:
            allen_clusters (List[List[List[int]]]): The coreference clusters from the AllenNLP model.
            document (List[str]): The document tokens as predicted by the AllenNLP model.
            doc (Doc): The SpaCy document containing the text and the Huggingface coreference clusters.
            hugging_clusters (List[List[List[int]]]): The coreference clusters from the Huggingface model.

        """
        allen_prediction = self.allen_model.predict(text)
        self.allen_clusters = allen_prediction["clusters"]
        self.document = allen_prediction["document"]
        self.doc = self.hugging_model(text)
        hugging_clusters = self._transform_huggingface_answer_to_allen_list_of_clusters()
        self.hugging_clusters = hugging_clusters

    def _transform_huggingface_answer_to_allen_list_of_clusters(self):
        """Transform the Huggingface answer format to the AllenNLP list of clusters format.

        Args:
            None

        Returns:
            List[List[List[int]]]: A list of coreference clusters, where each cluster is a list of spans,
                                and each span is a list of two integers representing the start and end
                                token indices.

        """
        list_of_clusters = []
        for cluster in self.doc._.coref_clusters:
            list_of_clusters.append([])
            for span in cluster:
                list_of_clusters[-1].append([span[0].i, span[-1].i])
        return list_of_clusters


class PartialIntersectionStrategy(IntersectionStrategy):
    """A strategy for partially intersecting coreference resolution clusters.

    This strategy identifies clusters from both models that have at least two spans in common
    and combines them into intersected clusters.
    """

    def get_intersected_clusters(self) -> List[List[List[int]]]:
        """Get the intersected clusters between the AllenNLP and Huggingface coreference resolution models.

        Intersected clusters are clusters from both models that have at least two spans in common.

        Returns:
            List[List[List[int]]]: A list of intersected clusters, where each cluster is a list of spans,
                                and each span is a list of two integers representing the start and end
                                token indices.

        """
        intersected_clusters: List[List[List[int]]] = []
        for allen_cluster in self.allen_clusters:
            intersected_cluster: List[List[int]] = []
            for hugging_cluster in self.hugging_clusters:
                allen_set = {tuple(span) for span in allen_cluster}
                hugging_set = {tuple(span) for span in hugging_cluster}
                intersect = sorted([list(el) for el in allen_set.intersection(hugging_set)])
                if len(intersect) > 1:
                    intersected_cluster += intersect
            if intersected_cluster:
                intersected_clusters.append(intersected_cluster)
        return intersected_clusters


class FuzzyIntersectionStrategy(PartialIntersectionStrategy):
    """Is treated as a PartialIntersectionStrategy, yet first must map AllenNLP spans and Huggingface spans."""

    @staticmethod
    def flatten_cluster(list_of_clusters: List[List[List[int]]]) -> List[List[int]]:
        """Flatten a list of clusters into a single list of spans.

        Args:
            list_of_clusters (List[List[List[int]]]): A list of clusters, where each cluster is a list of spans, and each span is a list of two integers describing the start and end of the span.

        Returns:
            List[List[int]]: A list of all the spans in the input clusters.

        Raises:
            ValueError: If the input list_of_clusters is None or empty.

        """
        if list_of_clusters is None or len(list_of_clusters) == 0:
            msg = "Input list_of_clusters is None or empty."
            raise ValueError(msg)
        return [span for cluster in list_of_clusters for span in cluster]

    def _check_whether_spans_are_within_range(self, allen_span: List[int], hugging_span: List[int]) -> bool:
        """Check whether two spans are within each other's range.

        Args:
            allen_span (List[int]): A span from the AllenNLP model, where the first element is the start token index and the second element is the end token index.
            hugging_span (List[int]): A span from the Huggingface model, where the first element is the start token index and the second element is the end token index.

        Returns:
            bool: Whether the two spans are within each other's range.

        """
        allen_range = range(allen_span[0], allen_span[1] + 1)
        hugging_range = range(hugging_span[0], hugging_span[1] + 1)
        allen_within = allen_span[0] in hugging_range and allen_span[1] in hugging_range
        hugging_within = hugging_span[0] in allen_range and hugging_span[1] in allen_range
        return allen_within or hugging_within

    def _add_span_to_list_dict(self, allen_span: List[int], hugging_span: List[int]) -> None:
        """Add a span pair to the swap dictionary based on their lengths.

        This function compares the lengths of two spans, one from the AllenNLP model
        and one from the Huggingface model, and adds the span pair to the swap
        dictionary such that the longer span is used as the key.

        Args:
            allen_span (List[int]): The span from the AllenNLP model, represented as a
                list of two integers indicating the start and end token indices.
            hugging_span (List[int]): The span from the Huggingface model, represented as
                a list of two integers indicating the start and end token indices.

        Returns:
            None

        """
        if allen_span[1] - allen_span[0] > hugging_span[1] - hugging_span[0]:
            self._add_element(allen_span, hugging_span)
        else:
            self._add_element(hugging_span, allen_span)

    def _add_element(self, key_span: List[int], val_span: List[int]) -> None:
        """Add a span to the swap dictionary.

        This function adds a span pair to the swap dictionary such that the key is
        the longer span and the value is the shorter span.

        Args:
            key_span (List[int]): The span to be used as the key in the swap dictionary.
            val_span (List[int]): The span to be used as the value in the swap dictionary.

        Returns:
            None

        """
        if tuple(key_span) in self.swap_dict_list:
            self.swap_dict_list[tuple(key_span)].append(tuple(val_span))
        else:
            self.swap_dict_list[tuple(key_span)] = [tuple(val_span)]

    def _filter_out_swap_dict(self) -> dict:
        """Filter out the swap dictionary so that each key maps to a single value span.

        This function takes the swap dictionary and filters out the values so that each
        key maps to a single value span. The value span is chosen as the longest span
        that is within the range of the key span.

        Returns:
            dict: A dictionary where each key maps to a single value span.

        """
        swap_dict = {}
        for key, vals in self.swap_dict_list.items():
            if self.swap_dict_list[key] != vals[0]:
                swap_dict[key] = sorted(vals, key=lambda x: x[1] - x[0], reverse=True)[0]
        return swap_dict

    def _swap_mapped_spans(self, list_of_clusters: List[List[List[int]]], model_dict: dict) -> List[List[List[int]]]:
        """Swap mapped spans in a list of clusters.

        This function takes a list of clusters and a swap dictionary and swaps out the
        spans in the list of clusters with the mapped spans in the swap dictionary.

        Args:
            list_of_clusters (List[List[List[int]]]): A list of clusters, where each cluster
                is a list of spans and each span is a list of two integers representing the
                start and end token indices.
            model_dict (dict): A dictionary where the keys are the spans to be swapped out
                and the values are the spans to swap in.

        Returns:
            List[List[List[int]]]: The list of clusters with the mapped spans swapped in.

        """
        for cluster_idx, cluster in enumerate(list_of_clusters):
            for span_idx, span in enumerate(cluster):
                if tuple(span) in model_dict:
                    list_of_clusters[cluster_idx][span_idx] = list(model_dict[tuple(span)])
        return list_of_clusters

    def get_mapped_spans_in_lists_of_clusters(self) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """Get mapped spans in lists of clusters for AllenNLP and Huggingface coreference models.

        This function takes the lists of clusters from the AllenNLP and Huggingface coreference models
        and maps the spans in the clusters such that the spans in the clusters are the same for both
        models. The mapping is done by comparing the spans in the clusters and choosing the longest
        span that is within the range of the other span.

        Returns:
            tuple: A tuple containing the mapped lists of clusters for the AllenNLP and Huggingface
                coreference models.

        """
        self.swap_dict_list = {}
        for allen_span in self.flatten_cluster(self.allen_clusters):
            for hugging_span in self.flatten_cluster(self.hugging_clusters):
                if self._check_whether_spans_are_within_range(allen_span, hugging_span):
                    self._add_span_to_list_dict(allen_span, hugging_span)
        swap_dict = self._filter_out_swap_dict()

        allen_clusters_mapped = self._swap_mapped_spans(deepcopy(self.allen_clusters), swap_dict)
        hugging_clusters_mapped = self._swap_mapped_spans(deepcopy(self.hugging_clusters), swap_dict)
        return allen_clusters_mapped, hugging_clusters_mapped

    def get_intersected_clusters(self) -> List[List[List[int]]]:
        """Get the intersected clusters between the AllenNLP and Huggingface coreference resolution models.

        This function first maps the spans in the clusters from both models to the same spans,
        and then calls the `get_intersected_clusters` method of the parent class to get the
        intersected clusters.

        Returns:
            List[List[List[int]]]: Intersected clusters between AllenNLP and Huggingface coreference
                resolution models, where each inner list is a cluster containing spans, and each
                span is a list of two integers representing the start and end token indices of the
                span.

        """
        (
            allen_clusters_mapped,
            hugging_clusters_mapped,
        ) = self.get_mapped_spans_in_lists_of_clusters()
        self.allen_clusters = allen_clusters_mapped
        self.hugging_clusters = hugging_clusters_mapped
        return super().get_intersected_clusters()


class StrictIntersectionStrategy(IntersectionStrategy):
    """A strategy for strictly intersecting coreference resolution clusters.

    This strategy identifies clusters that are exactly the same in both models
    and combines them into intersected clusters.
    """

    def get_intersected_clusters(self) -> List[List[List[int]]]:
        """Get the intersected clusters between the AllenNLP and Huggingface coreference resolution models.

        Strict intersection is defined as the clusters that are exactly the same in both models.

        Returns:
            List[List[List[int]]]: Intersected clusters between AllenNLP and Huggingface coreference
                resolution models, where each inner list is a cluster containing spans, and each
                span is a list of two integers representing the start and end token indices of the
                span.

        """
        intersected_clusters = []
        for allen_cluster in self.allen_clusters:
            for hugging_cluster in self.hugging_clusters:
                if allen_cluster == hugging_cluster:
                    intersected_clusters.extend([allen_cluster])
        return intersected_clusters


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


def print_clusters(doc: Doc, clusters: List[List[List[int]]]) -> None:
    """Print the coreference clusters in a human-readable format.

    Args:
        doc (Doc): The SpaCy document containing the text.
        clusters (List[List[List[int]]]): A list of coreference clusters, where each cluster is a list of spans,
                                          and each span is a list of two integers representing the start and end
                                          token indices.

    Returns:
        None

    """

    def get_span_words(span: List[int], allen_document: List[str]) -> str:
        """Get the words in a span of a document.

        Args:
            span (List[int]): A list of two integers representing the start and end token indices of the span.
            allen_document (List[str]): The list of words in the document.

        Returns:
            str: The words in the span, joined by spaces.

        """
        return " ".join(allen_document[span[0] : span[1] + 1])

    allen_document = [t.text for t in doc]
    for cluster in clusters:
        cluster_head_idx = get_cluster_head_idx(doc, cluster)
        if cluster_head_idx >= 0:
            cluster_head = cluster[cluster_head_idx]
            print(get_span_words(cluster_head, allen_document) + " - ", end="")  # noqa: T201
            print("[", end="")  # noqa: T201
            for i, span in enumerate(cluster):
                print(  # noqa: T201
                    get_span_words(span, allen_document) + ("; " if i + 1 < len(cluster) else ""),
                    end="",
                )
            print("]")  # noqa: T201
