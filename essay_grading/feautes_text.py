from __future__ import annotations

import pprint
import re
import time
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import spacy
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_extraction.text")

# import spacy
# from collections import defaultdict

# # Load the spaCy model (make sure you have it downloaded)
# # python -m spacy download en_core_web_sm
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     print("Downloading 'en_core_web_sm' model...")
#     spacy.cli.download("en_core_web_sm")
#     nlp = spacy.load("en_core_web_sm")


# # Define two example sentences
# sentence1 = "The black cat sat calmly on the mat."
# sentence2 = "A white kitten sits near the mat."
# sentence3 = "The dog chased the cat quickly."
# sentence4 = "The cat was chased by the dog." # Passive voice, different structure

# # Process the sentences with spaCy
# doc1 = nlp(sentence1)
# doc2 = nlp(sentence2)
# doc3 = nlp(sentence3)
# doc4 = nlp(sentence4)


# # Extract predicate-argument structures (improved version)
# def extract_predicate_arguments_improved(doc):
#     """
#     Extracts verbs as predicates and their key syntactic arguments (subj, obj).
#     Returns a dictionary mapping predicate lemma to a list of (dep_relation, argument_lemma) tuples.
#     """
#     predicate_arguments = defaultdict(list)
#     for token in doc:
#         # Consider all verbs as potential predicates
#         if token.pos_ == "VERB":
#             predicate_lemma = token.lemma_
#             arguments = []
#             for child in token.children:
#                 # Focus on core grammatical relations
#                 # nsubj: nominal subject, dobj: direct object,
#                 # nsubjpass: nominal subject (passive), auxpass: passive auxiliary (identifies passive)
#                 # We could add more like 'pobj' for prepositional objects if needed.
#                 if child.dep_ in ("nsubj", "dobj", "nsubjpass", "agent", "attr", "acomp", "xcomp"): # Added agent for passive, attr/acomp/xcomp for linking verbs/complements
#                      # Store the dependency relation and the argument's lemma
#                      arguments.append((child.dep_, child.lemma_))

#             # Handle passive voice slightly better (agent often attached to auxpass or verb)
#             if any(c.dep_ == 'auxpass' for c in token.children):
#                  for child in token.children:
#                      if child.dep_ == 'agent':
#                          for grand_child in child.children:
#                              if grand_child.dep_ == 'pobj': # Object of the 'by' preposition in agent phrase
#                                  arguments.append(('agent_pobj', grand_child.lemma_))


#             if arguments: # Only add if we found relevant arguments
#                 predicate_arguments[predicate_lemma].extend(arguments)

#     # Convert defaultdict back to dict for clarity if desired (optional)
#     return dict(predicate_arguments)


# # Calculate SRL similarity (improved version)
# def srl_similarity_improved(doc1, doc2):
#     srl_similarity_score = 0.0
#     match_count = 0

#     srl1 = extract_predicate_arguments_improved(doc1)
#     srl2 = extract_predicate_arguments_improved(doc2)

#     # Use predicate lemmas as keys
#     predicates1 = set(srl1.keys())
#     predicates2 = set(srl2.keys())

#     common_predicates = predicates1.intersection(predicates2)

#     if not common_predicates:
#         return 0.0

#     for predicate_lemma in common_predicates:
#         # Get argument lists for this common predicate
#         # srl1[predicate_lemma] is like [('nsubj', 'cat'), ('prep', 'on'), ...] but only has key deps
#         args1_list = srl1[predicate_lemma] # List of (dep, lemma) tuples
#         args2_list = srl2[predicate_lemma]

#         # Create sets of (dep, lemma) for easier comparison
#         args1_set = set(args1_list)
#         args2_set = set(args2_list)

#         # --- Similarity Calculation ---
#         # Option 1: Simple Jaccard on (dep, lemma) pairs
#         intersection_args = len(args1_set.intersection(args2_set))
#         union_args = len(args1_set.union(args2_set))
#         if union_args > 0:
#             similarity = intersection_args / union_args
#         else:
#             similarity = 1.0 if not args1_list and not args2_list else 0.0 # Both empty -> perfect match? Or 0? Let's say 1.

#         # Option 2 (More nuanced): Score based on lemma matches, boosted by role match
#         # similarity = 0
#         # arg1_lemmas = {lemma for dep, lemma in args1_list}
#         # arg2_lemmas = {lemma for dep, lemma in args2_list}
#         # common_lemmas = arg1_lemmas.intersection(arg2_lemmas)
#         # total_unique_lemmas = len(arg1_lemmas.union(arg2_lemmas))
#         # if total_unique_lemmas > 0:
#         #      base_sim = len(common_lemmas) / total_unique_lemmas
#         #      role_bonus = 0
#         #      matches_with_role = 0
#         #      for dep1, lemma1 in args1_list:
#         #           for dep2, lemma2 in args2_list:
#         #                if lemma1 == lemma2: # Lemma match
#         #                     if dep1 == dep2: # Role match
#         #                          matches_with_role += 1
#         #      # Simple bonus (could be more sophisticated)
#         #      if len(common_lemmas) > 0:
#         #           role_bonus = (matches_with_role / len(common_lemmas)) * 0.2 # Small bonus for role match
#         #      similarity = base_sim + role_bonus
#         # else:
#         #      similarity = 1.0 if not args1_list and not args2_list else 0.0
#         # similarity = min(similarity, 1.0) # Ensure score doesn't exceed 1

#         srl_similarity_score += similarity
#         match_count += 1


#     # Normalize score: Average similarity over the number of *matching* predicates
#     if match_count > 0:
#          normalized_score = srl_similarity_score / match_count
#     else:
#          normalized_score = 0.0 # No common predicates found

#     # Alternative normalization: Divide by total unique predicates?
#     # total_predicates = len(predicates1.union(predicates2))
#     # if total_predicates > 0:
#     #      normalized_score = srl_similarity_score / total_predicates
#     # else:
#     #      normalized_score = 1.0 if not predicates1 and not predicates2 else 0.0


#     return normalized_score


# # Calculate SRL similarity between the sentences
# srl_sim_12 = srl_similarity_improved(doc1, doc2)
# srl_sim_34 = srl_similarity_improved(doc3, doc4)
# srl_sim_13 = srl_similarity_improved(doc1, doc3)

# print(f"SRL Similarity (Sentence 1 vs 2): {srl_sim_12:.4f}")
# print(f"SRL Similarity (Sentence 3 vs 4): {srl_sim_34:.4f}") # Expect higher score due to lemma match + passive handling
# print(f"SRL Similarity (Sentence 1 vs 3): {srl_sim_13:.4f}") # Expect lower score

# # Optional: Print extracted structures to debug
# print("\nExtracted SRL for Sentence 1:", extract_predicate_arguments_improved(doc1))
# print("Extracted SRL for Sentence 2:", extract_predicate_arguments_improved(doc2))
# print("Extracted SRL for Sentence 3:", extract_predicate_arguments_improved(doc3))
# print("Extracted SRL for Sentence 4:", extract_predicate_arguments_improved(doc4))

s1 = "Education is the passport to the future, for tomorrow belongs to those who prepare for it today."
s2 = "The future belongs to those who prepare for it today; education is their passport."


# Improved tokenization
def simple_tokenize(text):
    if not isinstance(text, str):  # Handle potential non-string input
        return []
    text = text.lower()  # Lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    # Remove empty strings that might result from multiple spaces
    return [token for token in text.split() if token]


# Step 1: Create word vectors (using improved tokenizer)
def create_word_vectors(texts):
    # Use the tokenizer in CountVectorizer
    vectorizer = CountVectorizer(tokenizer=simple_tokenize)
    # Ensure texts are valid strings
    valid_texts = [t for t in texts if isinstance(t, str)]
    if not valid_texts:
        # Handle case where no valid texts are provided
        return None, []
    word_matrix = vectorizer.fit_transform(valid_texts)
    words = vectorizer.get_feature_names_out()
    return word_matrix, words


# Step 2: Build graph efficiently
def build_graph_efficiently(word_matrix, words):
    if word_matrix is None or len(words) == 0:
        return nx.Graph()  # Return empty graph if no data

    start_time = time.time()
    print(f"Building graph for {len(words)} words...")

    # Need word vectors (columns as rows)
    # Making dense for cosine_similarity. BEWARE of memory for large vocab/docs.
    word_vectors = word_matrix.T.toarray()

    # Calculate pairwise cosine similarity between word vectors
    similarity_matrix = cosine_similarity(word_vectors)  # Shape (num_words, num_words)
    print(f"  Similarity matrix calculation took {time.time() - start_time:.2f}s")

    graph_build_start = time.time()
    num_words = len(words)
    graph = nx.Graph()
    graph.add_nodes_from(words)  # Add all words as nodes

    # Use a threshold slightly above zero to avoid tiny similarities
    similarity_threshold = 0.01
    edges_added = 0

    # Iterate through the upper triangle of the similarity matrix
    for i in range(num_words):
        for j in range(i + 1, num_words):
            similarity = similarity_matrix[i, j]
            if similarity > similarity_threshold:
                graph.add_edge(words[i], words[j], weight=similarity)
                edges_added += 1

    print(f"  Graph construction (adding {edges_added} edges) took {time.time() - graph_build_start:.2f}s")
    print(f"Graph building finished. Total time: {time.time() - start_time:.2f}s")
    return graph


# Step 3: Calculate graph similarity using subgraph density
def calculate_graph_similarity(graph, text1, text2):
    words1 = set(simple_tokenize(text1))
    words2 = set(simple_tokenize(text2))

    # Find common words that are actually *in the graph* (part of the initial corpus vocab)
    common_words_in_graph = words1.intersection(words2).intersection(graph.nodes)

    if not common_words_in_graph:
        print("No common words found in the graph.")
        return 0.0

    # Important: Create subgraph only from nodes present in the main graph
    subgraph = graph.subgraph(common_words_in_graph)

    # Density calculation needs at least 2 nodes for potential edges
    if subgraph.number_of_nodes() < 2:
        print(f"Subgraph has < 2 nodes ({subgraph.number_of_nodes()}), density is undefined or 0.")
        # Density is often considered 0 for single nodes or empty graphs.
        # Check nx.density documentation for specific behavior if needed.
        return 0.0  # Or handle as appropriate

    try:
        graph_sim = nx.density(subgraph)
        print(
            f"  Subgraph Nodes: {subgraph.number_of_nodes()}, Edges: {subgraph.number_of_edges()}, Density: {graph_sim:.4f}"
        )
    except ZeroDivisionError:
        # This might happen if number_of_nodes is < 2, though checked above
        print("  Density calculation failed (ZeroDivisionError).")
        graph_sim = 0.0

    return graph_sim


# --- Example Usage ---
# Use a slightly larger corpus to build a more meaningful (but still limited) graph
corpus = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog jumps over a lazy fox",
    "The dog barks loudly at the fox",
    "Brown foxes are quick animals",
    "Never jump over the lazy dog quickly",
]

print("Step 1: Creating word vectors...")
word_matrix, words = create_word_vectors(corpus)

print("\nStep 2: Building graph...")
if word_matrix is not None:
    graph = build_graph_efficiently(word_matrix, words)
else:
    print("Could not create word matrix. Exiting.")
    exit()

print(f"\nGraph nodes: {graph.number_of_nodes()}, Graph edges: {graph.number_of_edges()}")


print("\nStep 3: Calculating similarity...")
text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A quick brown dog jumps over a lazy fox"
print(f"\nComparing:\n Text 1: '{text1}'\n Text 2: '{text2}'")
graph_similarity12 = calculate_graph_similarity(graph, text1, text2)
print(f"Graph Similarity (1 vs 2): {graph_similarity12:.4f}")

text3 = "the lazy fox barks at the dog"
print(f"\nComparing:\n Text 1: '{text1}'\n Text 3: '{text3}'")
graph_similarity13 = calculate_graph_similarity(graph, text1, text3)
print(f"Graph Similarity (1 vs 3): {graph_similarity13:.4f}")

text4 = "a slow red turtle"  # No common words in graph
print(f"\nComparing:\n Text 1: '{text1}'\n Text 4: '{text4}'")
graph_similarity14 = calculate_graph_similarity(graph, text1, text4)
print(f"Graph Similarity (1 vs 4): {graph_similarity14:.4f}")


def preprocess(text: str, lowercase: bool = True, remove_punct: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def build_ngram_index(tokens: List[str], k: int) -> Dict[Tuple[str, ...], List[int]]:
    """Map each k-gram to the list of start positions where it occurs."""
    index = defaultdict(list)
    for i in range(len(tokens) - k + 1):
        key = tuple(tokens[i : i + k])
        index[key].append(i)
    return index


def smith_waterman_window(
    t1: List[str],
    t2: List[str],
    sm: Dict[str, Dict[str, float]],
    gap_penalty: float,
    win1: Tuple[int, int],
    win2: Tuple[int, int],
) -> float:
    """Run SW on t1[w1[0]:w1[1]] vs t2[w2[0]:w2[1]] and return max score."""
    a1 = t1[win1[0] : win1[1]]
    a2 = t2[win2[0] : win2[1]]
    n, m = len(a1), len(a2)
    h_zeroes = np.zeros((n + 1, m + 1))
    best = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sc = sm[a1[i - 1]][a2[j - 1]]
            h_zeroes[i, j] = max(
                0,
                h_zeroes[i - 1, j - 1] + sc,
                h_zeroes[i - 1, j] + gap_penalty,
                h_zeroes[i, j - 1] + gap_penalty,
            )
            best = max(best, h_zeroes[i, j])
    return best


def compute_plagiarism_score_fast(
    text1: str,
    text2: str,
    k: int = 3,
    window_radius: int = 50,
    match_score: float = 1.0,
    mismatch_score: float = 0.0,
    gap_penalty: float = -1.0,
) -> float:
    # 1. Preprocess & tokenize
    t1 = preprocess(text1)
    t2 = preprocess(text2)

    # 2. Build scoring matrix once
    uniq = set(t1) | set(t2)
    sm = {w: {v: (match_score if w == v else mismatch_score) for v in uniq} for w in uniq}

    # 3. Build k-gram index on text2
    idx2 = build_ngram_index(t2, k)

    # 4. For each k-gram in text1 that appears in text2, extend around it
    best_score = 0.0
    for i in range(len(t1) - k + 1):
        gram = tuple(t1[i : i + k])
        for j in idx2.get(gram, []):
            # define window bounds, clipped to sequence ends
            w1_start = max(0, i - window_radius)
            w1_end = min(len(t1), i + k + window_radius)
            w2_start = max(0, j - window_radius)
            w2_end = min(len(t2), j + k + window_radius)

            score = smith_waterman_window(
                t1,
                t2,
                sm,
                gap_penalty,
                (w1_start, w1_end),
                (w2_start, w2_end),
            )
            best_score = max(best_score, score)

    # 5. Normalize by the smaller token count
    denom = min(len(t1), len(t2)) or 1
    return best_score / denom


pct = compute_plagiarism_score_fast(s1, s2)
print(f"Fast plagiarism overlap: {pct:.2%}")


# Step 1: Tokenize the texts
def tokenize_text(text):
    tokens = text.lower().split()  # Split the text into lowercase tokens
    return set(tokens)


# Step 2: Calculate overlap coefficient
def calculate_overlap_coefficient(text1, text2):
    set1 = tokenize_text(text1)
    set2 = tokenize_text(text2)
    intersection = len(set1.intersection(set2))
    min_size = min(len(set1), len(set2))
    overlap_coefficient = intersection / min_size if min_size > 0 else 0.0
    return overlap_coefficient


# Example usage
overlap_coefficient = calculate_overlap_coefficient(s1, s2)

print("Overlap Coefficient:", overlap_coefficient)


# Step 1: Tokenize the texts
def tokenize_text(text):
    tokens = text.lower().split()  # Split the text into lowercase tokens
    return set(tokens)


# Step 2: Calculate Sørensen-Dice coefficient
def calculate_sorensen_dice_coefficient(text1, text2):
    set1 = tokenize_text(text1)
    set2 = tokenize_text(text2)

    intersection = len(set1 & set2)
    total_tokens = len(set1) + len(set2)

    return ((2 * intersection) / total_tokens) if total_tokens > 0 else 0.0


dice_coefficient = calculate_sorensen_dice_coefficient(s1, s2)

print("Sørensen-Dice Coefficient:", dice_coefficient)


def get_char_by_char_equality_optimized(s1: Optional[str], s2: Optional[str]) -> float:
    """Compare two strings character by character with geometrically decaying weights, optimized for speed.

    Args:
        s1 (Optional[str]): The first string to compare.
        s2 (Optional[str]): The second string to compare.

    Returns:
        float: A similarity score between 0.0 and 1.0 based on character matches.
               Higher scores indicate greater similarity.

    Notes:
        - Handles None inputs by returning 0.0.
        - Comparison stops at the end of the shorter string.
        - Matches at the beginning contribute more (weights: 1.0, 0.5, 0.25, ...).

    """
    if s1 is None or s2 is None:
        return 0.0

    s1 = str(s1)
    s2 = str(s2)

    min_len = min(len(s1), len(s2))
    total_score = 0.0
    current_weight = 1.0

    for i in range(min_len):
        if s1[i] == s2[i]:
            total_score += current_weight
        current_weight *= 0.5

    return total_score


print("Character by character equality score:", get_char_by_char_equality_optimized(s1, s2))


# Load the spaCy model
nlp = spacy.load("en_core_web_sm")


# Step 1: Create semantic graph for text
def create_semantic_graph(text):
    doc = nlp(text)
    graph = nx.Graph()
    for token in doc:
        # Add nodes for tokens
        graph.add_node(token.i, text=token.text, lemma=token.lemma_, pos=token.pos_)
        # Add edges between tokens and their dependencies
        for child in token.children:
            graph.add_edge(token.i, child.i, label=child.dep_)
    return graph


# Step 2: Calculate similarity between semantic graphs
def calculate_graph_similarity(graph1, graph2):
    # Extract sets of nodes and edges from the graphs
    nodes1 = set(graph1.nodes)
    nodes2 = set(graph2.nodes)
    edges1 = set(graph1.edges)
    edges2 = set(graph2.edges)
    # Calculate Jaccard similarity coefficient
    jaccard_similarity_nodes = len(nodes1.intersection(nodes2)) / len(nodes1.union(nodes2))
    jaccard_similarity_edges = len(edges1.intersection(edges2)) / len(edges1.union(edges2))
    # Combine node and edge similarities using a weighted sum or another method
    # Here, we'll use a simple average
    jaccard_similarity = (jaccard_similarity_nodes + jaccard_similarity_edges) / 2
    return jaccard_similarity


# Example usage

# Step 1: Create semantic graphs for both texts
graph1 = create_semantic_graph(s1)
graph2 = create_semantic_graph(s2)

# Step 2: Calculate similarity between semantic graphs
graph_similarity = calculate_graph_similarity(graph1, graph2)

print("Graph Similarity:", graph_similarity)


# --- Lexical/Clustering Features ---


def preprocess_tfidf(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    if lowercase:
        text = text.lower()
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    return text


def extract_lexical_features(
    model_answers: List[str],
    student_answers: List[str],
    linkage_method: str = "average",
    distance_metric: str = "sqeuclidean",
    cluster_dist_thresh: float = 0.5,
) -> Dict[str, float]:
    # 1. Build the *global* corpus
    all_texts = [preprocess_tfidf(t) for t in model_answers + student_answers]
    n_models = len(model_answers)
    n_total = len(all_texts)

    # 2. TF-IDF encode once
    vec = TfidfVectorizer()
    X = vec.fit_transform(all_texts).toarray()

    # 3. Pairwise cosine similarity and cophenetic
    D = pdist(X, metric=distance_metric)
    # distm = squareform(D)
    # simm = 1.0 - distm

    # 4. One linkage on entire set
    link = linkage(D, method=linkage_method)
    _, coph = cophenet(link, D)
    cophm = squareform(coph)

    # 5. One distance-threshold clustering
    labels = fcluster(link, t=cluster_dist_thresh, criterion="distance")
    # 6. One global silhouette (requires >1 cluster)
    sil = silhouette_samples(X, labels, metric=distance_metric) if len(np.unique(labels)) > 1 else np.zeros(n_total)

    features = []
    # Now extract per-student from start from n_models index till n_total global index which is students clusers
    for idx, _ in enumerate(student_answers):
        student_idx = n_models + idx
        # cosine to each model
        # sims_to_models = simm[student_idx, :n_models]
        # cophenetic to each model
        cops_to_models = cophm[student_idx, :n_models]

        lbl = labels[student_idx]
        size = int((labels == lbl).sum())

        feats = {
            # "cosine_min": float(sims_to_models.min()),
            # "cosine_mean": float(sims_to_models.mean()),
            # "cosine_max": float(sims_to_models.max()),
            "coph_min": float(cops_to_models.min()),
            "coph_mean": float(cops_to_models.mean()),
            "coph_max": float(cops_to_models.max()),
            "cluster_label": lbl,
            "cluster_size": size,
            "is_outlier": int(size == 1),
            "silhouette": float(sil[student_idx]),
            "index": student_idx,
            # "text": student_text,
        }
        features.append(feats)

    return features


# --- Combine, Normalize, Train ---


def build_feature_matrix(model_answers: List[str], student_texts: List[str]) -> np.ndarray:
    """Return raw feature matrix of shape (n_students, n_features)."""
    # 1) Compute batch lexical features for all students at once
    batch_feats = extract_lexical_features(model_answers, student_texts)
    return batch_feats
    # 2) Compute Smith  Waterman overlap per student
    # sw_feats = [max(compute_plagiarism_score(text, m) for m in model_answers) for text in student_texts]

    # 3) Merge into a feature matrix
    # feature_names = ["sw_overlap", *list(batch_feats[0].keys())]
    # X = []
    # for sw, lex in zip(sw_feats, batch_feats):
    #     row = [sw] + [lex[k] for k in lex]
    #     X.append(row)
    # return np.array(X), feature_names


def normalize_matrix(X: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(X)


if __name__ == "__main__":
    # --- Example data ---
    model_answers = [
        "Education is the passport to the future, for tomorrow belongs to those who prepare for it today.",
        "The future belongs to those who prepare for it today; education is their passport.",
    ]
    # Student essays and their human scores (e.g., 0–100)
    student_texts = [
        "Tomorrow belongs to those who plan ahead; learning opens doors to tomorrow.",
        "Education is key to the future because those who learn early succeed.",
        "Cooking recipes differ from studying methods for tomorrow's success.",  # lower relevance,
        "The future belongs to those who prepare for it today; education is their passport.",
    ]
    human_scores = np.array([85, 90, 40, 100])

    # Build & normalize features
    X_raw = build_feature_matrix(model_answers, student_texts)
    pprint.pprint(f"Raw feature matrix:\n{X_raw}")
    # X = normalize_matrix(X_raw)

    # # Train & cross-validate a Ridge regressor
    # model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3)
    # cv_scores = cross_val_score(model, X, human_scores, cv=3, scoring="r2")
    # print(f"Features: {feature_names}")
    # print(f"CV R² scores: {cv_scores}")
    # model.fit(X, human_scores)

    # # Example prediction
    # new_essay = "Those who study today will own the future because education is the key."
    # X_new, _ = build_feature_matrix(model_answers, [new_essay])
    # X_new = normalize_matrix(np.vstack([X_raw, X_new]))[-1].reshape(1, -1)
    # pred = model.predict(X_new)[0]
    # print(f"Predicted score for new essay: {pred:.1f}")


# ------------------------------------------------------------------------------
# Rule-Based Coreference Resolution using SpaCy
# ------------------------------------------------------------------------------
# This script implements a rule-based approach to coreference resolution,
# a critical task in Natural Language Processing (NLP) for identifying
# expressions in text that refer to the same entity (persons, places, things, etc.).
#
# Approach Inspired By:
# - The concepts outlined in general NLP coreference resolution literature.
# - Utilizes SpaCy (en_core_web_md) for linguistic features:
#   - Part-of-Speech (POS) tagging
#   - Dependency Parsing (Syntactic Structure)
#   - Named Entity Recognition (NER)
#   - Morphological Analysis (Number, Gender)
#   - Word Vectors (for semantic similarity fallback)
#
# Methodology (Rule-Based):
# As described in the article "The Key to Unlocking True Language Understanding:
# Coreference Resolution", this system relies on predefined linguistic rules
# and heuristics based on syntactic and semantic patterns. It processes
# potential referring expressions (mentions), primarily pronouns and proper nouns,
# and attempts to link them to preceding potential antecedents.
#
# Key Rules Implemented:
# 1. Pleonastic 'It' detection (non-referential 'it')
# 2. Reflexive pronoun resolution (e.g., 'himself' -> subject)
# 3. Relative pronoun resolution (e.g., 'who'/'which' -> head noun)
# 4. Quoted speech pronoun resolution ('I'/'we' -> speaker)
# 5. Possessive pronoun resolution ('his'/'her' -> possessor)
# 6. Standard pronoun resolution (backward search with agreement checks)
# 7. Proper noun matching (exact and partial matches)
#
# Scoring & Ambiguity Handling:
# To address the challenge of ambiguity mentioned in the article, rules are
# prioritized, and a confidence score is assigned based on the rule's reliability
# and contextual factors like NER matches, subject salience, and proximity.
# More reliable rules (e.g., Reflexive, Relative) get higher scores.
#
# Context Management:
# Resolving coreferences often depends on context. This system uses a
# configurable sentence window (`search_sentences`) to limit the search space
# for antecedents, balancing recall with the challenge of limited context.
#
# Limitations (Compared to ML/Neural Approaches):
# - Interpretability: Rule-based systems are generally more interpretable.
# - Complexity & Variability: May struggle with the vast complexity and variability
#   of natural language compared to models trained on large corpora.
# - Generalization: May not generalize as well to unseen patterns.
# - World Knowledge: Lacks deep common sense or world knowledge.
# - Definite Noun Phrase & Clausal Coreference: Primarily focuses on pronouns
#   and proper nouns, with limited handling of other referring expression types.
# - Cluster Building: Outputs pairs, not fully resolved entity clusters (though
#   pairs could be used for downstream clustering).
#
# Evaluation:
# Standard metrics like MUC, B-Cubed, and CoNLL F1 are typically used to evaluate
# coreference resolution systems, assessing performance on mentions, links, and chains.
# ------------------------------------------------------------------------------
import spacy
from spacy.tokens import Token, Doc, Span
import warnings  # To potentially filter UserWarnings from similarity

# --- Constants ---
COLLECTIVE_NOUNS = {"team", "committee", "government", "group", "company", "staff", "jury"}

# --- Load Model ---
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading en_core_web_md model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")


# --- Helper Functions (Unchanged from previous 'revised' version) ---
def get_gender(token: Token):
    gender = token.morph.get("Gender")
    if token.lemma_ in ("he", "his", "him", "himself"):
        return ["Masc"]
    if token.lemma_ in ("she", "her", "hers", "herself"):
        return ["Fem"]
    if token.lemma_ == "it":
        return ["Neut"]
    if token.ent_type_ == "PERSON":
        lower_text = token.text.lower()
        if lower_text in ["john", "paul", "mike", "peter", "bob", "james", "william", "david"]:
            return ["Masc"]
        if lower_text in ["mary", "lisa", "sarah", "alice", "susan", "evelyn", "jane", "elizabeth"]:
            return ["Fem"]
    return gender if gender else ["Neut"]


def get_number(token: Token):
    number = token.morph.get("Number")
    if token.lemma_ in ("he", "she", "it", "i", "me", "myself", "himself", "herself", "itself"):
        return ["Sing"]
    if token.lemma_ in ("we", "they", "us", "them", "ourselves", "themselves"):
        return ["Plur"]
    if not number:
        if token.tag_ in ["NN", "NNP"]:
            return ["Sing"]
        if token.tag_ in ["NNS", "NNPS"]:
            return ["Plur"]
    return number if number else ["Sing"]


def check_agreement(pronoun: Token, candidate: Token):
    pronoun_number = get_number(pronoun)
    candidate_number = get_number(candidate)
    pronoun_gender = get_gender(pronoun)
    candidate_gender = get_gender(candidate)
    number_mismatch = not set(pronoun_number).intersection(candidate_number)
    is_singular_they_case = False
    is_collective_noun_case = False
    if number_mismatch:
        is_singular_they_case = (
            "Sing" in candidate_number
            and pronoun.lemma_ == "they"
            and (candidate.ent_type_ == "PERSON" or candidate.lemma_ == "friend")
        )
        is_collective_noun_case = (
            "Plur" in pronoun_number
            and "Sing" in candidate_number
            and (candidate.ent_type_ == "ORG" or candidate.lemma_ in COLLECTIVE_NOUNS)
        )
        if not (is_singular_they_case or is_collective_noun_case):
            return False, False
    pron_gender_set = set(pronoun_gender)
    cand_gender_set = set(candidate_gender)
    if pronoun.tag_ in ["WP", "WDT"]:
        if pronoun.lemma_ == "who" and candidate.ent_type_ != "PERSON":
            return False, False
        if pronoun.lemma_ == "which" and candidate.ent_type_ == "PERSON":
            return False, False
    else:
        if "Neut" in pron_gender_set and candidate.ent_type_ == "PERSON":
            return False, False
        if candidate.ent_type_ == "PERSON":
            if ("Masc" in pron_gender_set and "Fem" in cand_gender_set) or (
                "Fem" in pron_gender_set and "Masc" in cand_gender_set
            ):
                return False, False
        elif candidate.ent_type_ != "PERSON":
            if ("Masc" in pron_gender_set and "Fem" in cand_gender_set) or (
                "Fem" in pron_gender_set and "Masc" in cand_gender_set
            ):
                return False, False
    return True, is_singular_they_case


def is_reflexive(token: Token):
    return token.tag_ == "PRP" and token.lemma_.endswith("self")


def find_subject(token: Token):
    head = token.head
    while head.pos_ not in ("VERB", "AUX") and head.head != head:
        head = head.head
    if head.pos_ in ("VERB", "AUX"):
        subjects = [child for child in head.children if child.dep_ in ("nsubj", "nsubjpass")]
        if subjects:
            core_subj = subjects[0]
            while core_subj.dep_ == "compound" and core_subj.head.i < token.i:
                core_subj = core_subj.head
            if core_subj.pos_ == "NOUN" and core_subj.ent_type_ != "PERSON":
                person_in_subj = [t for t in core_subj.subtree if t.ent_type_ == "PERSON"]
                if person_in_subj:
                    return person_in_subj[0]
            return core_subj
    return None


def get_sentence_span(doc: Doc, token_index: int) -> Span | None:
    token = doc[token_index]
    return token.sent


def find_speaker(pronoun: Token) -> Token | None:
    reporting_verbs = {"say", "tell", "ask", "reply", "shout", "whisper", "claim", "state"}
    head = pronoun.head
    in_quote_clause = False
    clause_verb = pronoun
    potential_report_verb = None
    while clause_verb.head != clause_verb and clause_verb.dep_ != "ROOT":
        potential_report_verb = clause_verb.head
        if potential_report_verb.lemma_ in reporting_verbs and clause_verb.dep_ in ("ccomp", "dobj", "advcl"):
            in_quote_clause = True
            break
        if head.lemma_ in reporting_verbs and pronoun.dep_ == "dobj":  # Direct object case
            in_quote_clause = True
            potential_report_verb = head
            break
        clause_verb = clause_verb.head
    if in_quote_clause and potential_report_verb:
        reporting_verb = potential_report_verb
        for child in reporting_verb.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                return child
    return None


def is_pleonastic_it(token: Token) -> bool:
    if token.lemma_ != "it":
        return False
    if token.dep_ == "nsubj":
        verb = token.head
        if verb.pos_ == "VERB":
            if verb.lemma_ in {"seem", "appear", "look", "sound", "feel", "happen", "matter", "turn out"}:
                if any(c.dep_ in ("ccomp", "csubj", "xcomp") for c in verb.children):
                    return True  # Simplified check
            if verb.lemma_ in {"rain", "snow", "hail", "thunder", "lighten"}:
                return True
        elif verb.lemma_ == "be" and verb.pos_ == "AUX":
            attr = next((child for child in verb.children if child.dep_ == "attr"), None)
            if attr:
                if attr.ent_type_ == "TIME" or any(t.ent_type_ == "TIME" for t in attr.subtree):
                    return True
                if any(t.text.lower() in ["o'clock", "pm", "am", "noon", "midnight"] for t in attr.subtree):
                    return True
                relcl = next((child for child in verb.children if child.dep_ == "relcl"), None)
                if relcl and relcl.head == attr:
                    return True  # Cleft
    return False


# --- End Helper Functions ---


# --- Main Resolution Function ---
def rule_based_coref_resolution_with_indices(  # Renamed function
    text: str, similarity_threshold: float = 0.5, use_similarity_fallback: bool = False, search_sentences: int = 2
):
    """
    Applies revised rule-based coreference resolution and returns results
    with character indices.

    Args:
        text (str): The input text document.
        similarity_threshold (float): Minimum word vector similarity for fallback rule.
        use_similarity_fallback (bool): Enable/disable similarity fallback rule.
        search_sentences (int): Number of sentences (current + preceding) to search for antecedents.

    Returns:
        list[tuple[dict, dict, float, str]]: A list of resolved coreference pairs.
            Each tuple contains:
            (
                {'text': str, 'start': int, 'end': int}, # Mention span info
                {'text': str, 'start': int, 'end': int}, # Antecedent span info
                float,                                  # Confidence score
                str                                     # Rule name
            )
    """
    doc = nlp(text)
    # Stores results internally as: (mention_token, antecedent_token, confidence_score, rule_name)
    coref_results_internal = []
    processed_mentions = set()

    # --- Processing Loop (Identical logic to 'revised' version) ---
    for sent_idx, sentence in enumerate(doc.sents):
        start_search_token_idx = 0
        if search_sentences > 1 and sent_idx > 0:
            sents_list = list(doc.sents)
            first_sent_idx_in_window = max(0, sent_idx - search_sentences + 1)
            start_search_token_idx = sents_list[first_sent_idx_in_window].start

        for i in range(sentence.start, sentence.end):
            token = doc[i]
            if token.i in processed_mentions:
                continue

            antecedent = None
            confidence = 0.0
            rule = "N/A"
            best_candidate_info = None

            # Pronoun Types
            is_personal_pronoun = token.tag_ == "PRP"
            is_possessive_pronoun = token.tag_ == "PRP$"
            is_relative_possessive = token.tag_ == "WP$"
            is_relative_nonpossessive = token.tag_ in ["WP", "WDT"]
            is_reflexive_pronoun = is_reflexive(token)

            if is_personal_pronoun or is_possessive_pronoun or is_relative_possessive or is_relative_nonpossessive:
                # Rule 0: Pleonastic 'It'
                if token.lemma_ == "it" and is_pleonastic_it(token):
                    processed_mentions.add(token.i)
                    continue
                # Rule 1: Reflexive
                if is_reflexive_pronoun:
                    subj = find_subject(token)
                    if subj:
                        antecedent = subj
                        confidence = 0.95
                        rule = "Reflexive Pronoun -> Subject"
                # Rule 2a: Relative Non-Possessive
                elif is_relative_nonpossessive:
                    potential_antecedent = token.head
                    if potential_antecedent.pos_ == "ADP":
                        potential_antecedent = potential_antecedent.head
                    if potential_antecedent.pos_ in {"NOUN", "PROPN", "PRON"}:
                        agrees, _ = check_agreement(token, potential_antecedent)
                        if agrees:
                            antecedent = potential_antecedent
                            confidence = 0.92
                            rule = "Relative Pronoun -> Syntactic Head"
                # Rule 3: Quoted Pronouns
                if not antecedent and token.lemma_ in {"i", "me", "my", "we", "us", "our"}:
                    speaker = find_speaker(token)
                    if speaker:
                        agrees, _ = check_agreement(token, speaker)
                        if agrees:
                            antecedent = speaker
                            confidence = 0.90
                            rule = "Quoted Pronoun -> Speaker"
                # Rule 2b / Rule 4: Possessive / Relative 'whose'
                elif not antecedent and (is_possessive_pronoun or is_relative_possessive):
                    # --- Backward Search Logic for Possessor ---
                    potential_candidates = []
                    search_rule_name = (
                        "Possessive Antecedent" if is_possessive_pronoun else "Relative Possessive (whose) Antecedent"
                    )
                    for j in range(token.i - 1, start_search_token_idx - 1, -1):
                        if j < 0:
                            break
                        candidate = doc[j]
                        if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                            continue
                        agrees, _ = check_agreement(token, candidate)
                        if not agrees:
                            continue
                        # Scoring...
                        cand_score = 0.05
                        cand_rule_detail = ""
                        if candidate.ent_type_ == "PERSON":
                            cand_score = max(cand_score, 0.75)
                        elif candidate.ent_type_:
                            cand_score = max(cand_score, 0.65)
                        if candidate.dep_ in ("nsubj", "nsubjpass"):
                            subject_bonus = 0.15
                            cand_score += subject_bonus
                            cand_rule_detail += " (Subject)"
                        distance = token.i - candidate.i
                        proximity_factor = max(0.1, 1.0 - (distance / 50.0))
                        cand_score *= proximity_factor
                        cand_score = min(cand_score, 1.0)
                        if cand_score > 0.05:
                            potential_candidates.append(
                                {
                                    "token": candidate,
                                    "score": cand_score,
                                    "reason": f"{search_rule_name}{cand_rule_detail}",
                                    "distance": distance,
                                }
                            )
                    if potential_candidates:
                        potential_candidates.sort(key=lambda x: (-x["score"], x["distance"]))
                        best_candidate_info = potential_candidates[0]
                # Rule 5: Standard Personal Pronouns
                elif not antecedent and is_personal_pronoun:
                    # --- Backward Search Logic for Standard Pronouns ---
                    potential_candidates = []
                    for j in range(token.i - 1, start_search_token_idx - 1, -1):
                        if j < 0:
                            break
                        candidate = doc[j]
                        if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                            continue
                        agrees, is_singular_they = check_agreement(token, candidate)
                        if not agrees:
                            continue
                        # Scoring...
                        cand_score = 0.01
                        cand_rule_detail = ""
                        if candidate.ent_type_:
                            if candidate.ent_type_ == "PERSON" and (token.lemma_ in ["he", "she", "they"]):
                                cand_score = max(cand_score, 0.70)
                                cand_rule_detail = "NER PERSON"
                            elif token.lemma_ == "it" and candidate.ent_type_ and candidate.ent_type_ != "PERSON":
                                cand_score = max(cand_score, 0.65)
                                cand_rule_detail = "NER Non-PERSON ('it')"
                        if is_singular_they and cand_score < 0.70:
                            cand_score = max(cand_score, 0.75)
                            cand_rule_detail = "Singular They Match"
                        if candidate.dep_ in ("nsubj", "nsubjpass"):
                            subject_bonus = 0.15
                            cand_score += subject_bonus
                            cand_rule_detail += " (Subject)" if cand_rule_detail else "Subject Salience"
                        distance = token.i - candidate.i
                        proximity_factor = max(0.1, 1.0 - (distance / 75.0))
                        cand_score *= proximity_factor
                        cand_score = min(cand_score, 1.0)
                        if use_similarity_fallback and cand_score < similarity_threshold:  # Similarity fallback...
                            try:
                                similarity = token.similarity(candidate)
                                if similarity >= similarity_threshold:
                                    sim_score = (
                                        0.1 + (similarity - similarity_threshold) / (1.0 - similarity_threshold) * 0.3
                                    )
                                    cand_score = max(cand_score, sim_score)
                                    cand_rule_detail = f"Similarity ({similarity:.2f})"
                            except UserWarning:
                                pass
                        if cand_score > 0.05:
                            potential_candidates.append(
                                {
                                    "token": candidate,
                                    "score": cand_score,
                                    "reason": f"Std Pronoun: {cand_rule_detail.strip()}",
                                    "distance": distance,
                                }
                            )
                    if potential_candidates:
                        potential_candidates.sort(key=lambda x: (-x["score"], x["distance"]))
                        best_candidate_info = potential_candidates[0]
                # Set antecedent from backward search if found
                if (
                    best_candidate_info and not antecedent
                ):  # Only if no specific rule (reflexive, relative, quote) found it first
                    antecedent = best_candidate_info["token"]
                    confidence = best_candidate_info["score"]
                    rule = best_candidate_info["reason"]

            # --- B. Proper Noun (PN) Coreference Logic ---
            elif token.pos_ == "PROPN":
                # --- PN Matching Logic ---
                potential_pn_antecedents = []
                for j in range(token.i - 1, start_search_token_idx - 1, -1):
                    if j < 0:
                        break
                    candidate = doc[j]
                    if candidate.pos_ == "PROPN" and candidate.ent_type_ == "PERSON":
                        if candidate.text == token.text:
                            potential_pn_antecedents.append(
                                {
                                    "token": candidate,
                                    "score": 0.98,
                                    "type": "Exact",
                                    "rule": "PN Exact Match",
                                    "distance": token.i - j,
                                }
                            )
                        candidate_is_longer = len(candidate.text.split()) > 1
                        token_is_shorter = len(token.text.split()) == 1
                        if candidate_is_longer and token_is_shorter and candidate.text.endswith(token.text):
                            prev_token = doc[token.i - 1] if token.i > 0 else None
                            likely_standalone = not (
                                prev_token and prev_token.pos_ == "PROPN" and prev_token.ent_iob_ != "O"
                            )
                            if likely_standalone:
                                potential_pn_antecedents.append(
                                    {
                                        "token": candidate,
                                        "score": 0.95,
                                        "type": "Partial",
                                        "rule": "PN Partial Match (Last)",
                                        "distance": token.i - j,
                                    }
                                )
                if potential_pn_antecedents:
                    potential_pn_antecedents.sort(key=lambda x: (-x["score"], x["distance"]))
                    best_pn_match = potential_pn_antecedents[0]
                    if best_pn_match["token"].i != token.i and best_pn_match["token"].i not in processed_mentions:
                        antecedent = best_pn_match["token"]
                        confidence = best_pn_match["score"]
                        rule = best_pn_match["rule"]

            # --- Store Result (Internal Token Format) ---
            if antecedent and antecedent.i != token.i:
                coref_results_internal.append((token, antecedent, round(confidence, 2), rule))
                processed_mentions.add(token.i)

    # --- Convert results to final output format with indices ---
    coref_pairs_with_indices = []
    for mention_tok, ant_tok, conf, rule_name in coref_results_internal:
        mention_span = {
            "text": mention_tok.text,
            "start": mention_tok.idx,
            "end": mention_tok.idx + len(mention_tok.text),
        }
        antecedent_span = {"text": ant_tok.text, "start": ant_tok.idx, "end": ant_tok.idx + len(ant_tok.text)}
        coref_pairs_with_indices.append((mention_span, antecedent_span, conf, rule_name))

    return coref_pairs_with_indices


# --- Testing ---
# [Include the same testing samples and loop as in the previous 'expert' version]
samples_expert = {
    "Pleonastic It": "It is raining heavily today. It seems that the game will be cancelled.",
    "Relative Who": "The man who arrived late missed the announcement.",  # who -> man
    "Relative Which": "The report, which detailed the findings, was released.",  # which -> report
    "Relative Whose": "The artist whose painting won the prize was ecstatic.",  # whose -> artist
    "Subject Salience": "The cat chased the mouse. It was fast.",  # It -> cat (subject likely preferred over mouse)
    "Possessive His": "John loves his dog.",  # his -> John
    "Possessive Its": "The company announced its profits.",  # its -> company
    "Quote Possessive": 'Mary said, "My car is blue."',  # My -> Mary
    "PN Partial Refined": "Professor John Smith presented. Later, Smith answered questions. Jane Smith watched.",  # Smith -> John Smith (not Jane Smith)
    "Complex Sentence": "Although the team lost, they showed great spirit, which pleased their coach.",  # they->team, which->spirit?, their->team
    "Weather/Time It": "It is snowing and it is almost noon.",
    "Cleft It": "It was Susan who solved the puzzle.",
}
samples_advanced = {
    "Appositive": "The CEO of the company, John, gave a speech. He emphasized the importance of innovation.",
    "Possessive": "John's car is red. It is fast.",  # It -> car
    "Definite Description": "The president gave a speech. He emphasized unity.",
    "Simple Morphology": "Sarah went to the market. She bought fruits.",
    "Mixed Case": "My friend Lisa arrived. She said that the party was fun. Lisa loves dancing.",
    "Plural": "The developers released the software. They were proud of it.",  # They -> developers, it -> software
    "Reflexive": "The manager told himself to stay calm.",
    "It ambiguity": "We poured water into the cup until it was full.",  # it -> cup
    "Singular They": "My friend mentioned their new job. They seem happy.",  # They -> friend
    "Complex": "Alice told Bob that she liked his new car, but he thought it was too flashy.",  # she->Alice, his->Bob(possessive), he->Bob, it->car
    "Quote Simple": 'Mary said, "I need coffee."',  # I -> Mary
    "Quote Complex": 'John asked his team, "Can we finish this today?" They replied affirmatively.',  # we -> team? John+team?, They -> team
    "Quote Nested": "The report stated, \"The witness claimed, 'He saw the suspect.'\" He later recanted.",  # He (inner) -> witness? suspect?, He (outer) -> witness?
    "Sentence Window": "Peter called Mike. He was happy. Later, Susan arrived. She brought cake.",  # He->Peter, She->Susan (test windowing)
    "Proper Noun Repeat": "Dr. Evelyn Reed published her findings. Reed argued for a new approach.",  # Reed -> Dr. Evelyn Reed
    "Proper Noun Partial": "Chairman John Smith entered. Smith looked tired.",  # Smith -> John Smith
}
all_samples = {**samples_advanced, **samples_expert}


print("--- Running Coreference Resolution with Documented Rules ---")
for description, text in all_samples.items():
    print(f"\n--- [{description}] ---")
    print(f"Text: {text}")
    try:
        # Using similarity=False to primarily test rule strength
        pairs = rule_based_coref_resolution_with_indices(text, search_sentences=2, use_similarity_fallback=True)
        print("Coreference pairs (Mention, Antecedent, Confidence, Rule):")
        if pairs:
            for pair in pairs:
                print(f"  - {pair[0]} -> {pair[1]} (Conf: {pair[2]}, Rule: {pair[3]})")
        else:
            print("  No pairs found.")

    except Exception as e:
        print(f"\n!!! An error occurred processing '{description}': {e} !!!")
        import traceback

        traceback.print_exc()
