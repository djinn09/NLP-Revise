
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