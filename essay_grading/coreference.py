from __future__ import annotations
import gender_guesser.detector as gender
import spacy
from spacy.tokens import Doc, Span, Token

# --- Constants ---
COLLECTIVE_NOUNS = {"team", "committee", "government", "group", "company", "staff", "jury", "class", "party"}
REPORTING_VERBS = {
    "say",
    "tell",
    "ask",
    "reply",
    "shout",
    "whisper",
    "claim",
    "state",
    "add",
    "explain",
    "note",
    "report",
    "argue",
}
PERSONAL_PRONOUN_LEMMAS = {"he", "she", "it", "they", "we", "i", "you"}
# List of common inanimate nouns likely to be Neut
NEUTER_NOUNS = {
    "car",
    "book",
    "table",
    "house",
    "report",
    "software",
    "cup",
    "water",
    "painting",
    "findings",
    "approach",
    "spirit",
    "puzzle",
    "job",
    "speech",
    "market",
    "party",
    "coffee",
}

# --- Load Model ---
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading en_core_web_md model...")
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

# --- Helper Functions (with Unspecified Gender Handling) ---

# 1. Pronoun → gender mapping
PRONOUN_GENDER = {
    **dict.fromkeys(["he", "his", "him", "himself"], "Masc"),
    **dict.fromkeys(["she", "her", "hers", "herself"], "Fem"),
    **dict.fromkeys(["it"], "Neut"),
}

# 2. Static name-based hints (small list of common names)
MASC_NAMES = {"john", "paul", "mike", "peter", "bob", "james", "william", "david", "george"}
FEM_NAMES = {"mary", "lisa", "sarah", "alice", "susan", "evelyn", "jane", "elizabeth", "ann", "kate"}
NAME_GENDER = dict.fromkeys(MASC_NAMES, "Masc")
NAME_GENDER.update(dict.fromkeys(FEM_NAMES, "Fem"))

# 3. Common neuter nouns (expandable list)
NEUTER_NOUNS = {"object", "device", "tool", "manager", "car", "book", "company"}

# 4. Initialize gender-guesser detector
DETECTOR = gender.Detector(case_sensitive=False)


def get_gender(token: Token):
    """Determine the 'gender' feature for a spacy Token.

    Order of checks:
      1. Pronoun lookup
      2. SpaCy morphological 'Gender'
      3. Static name lookup
      4. gender-guesser fallback for PERSON entities
      5. Common neuter nouns
      6. Default to 'Unspecified'.
    """
    # Normalize lemma and text
    lemma = token.lemma_.lower()
    text = token.text.strip().lower()
    # 1. Pronoun-based gender
    if lemma in PRONOUN_GENDER:
        return [PRONOUN_GENDER[lemma]]

    # 2. SpaCy morphological gender
    gender_feats = token.morph.get("Gender", None)
    if gender_feats:
        return gender_feats

    # 3. Named entity static lookup
    if token.ent_type_ == "PERSON":
        # Static name hints
        if text in NAME_GENDER:
            return [NAME_GENDER[text]]
        # 4. gender-guesser fallback
        guess = DETECTOR.get_gender(text)
        if guess in ("male", "mostly_male"):
            return ["Masc"]
        if guess in ("female", "mostly_female"):
            return ["Fem"]
        # PERSON but unknown -> unspecified
        return ["Unspecified"]

    # 5. Common neuter nouns
    if token.pos_ == "NOUN" and lemma in NEUTER_NOUNS:
        return ["Neut"]

    # 6. Default
    return ["Unspecified"]


# Pronoun → number mapping
PRONOUN_NUMBER = {
    **dict.fromkeys(["he", "she", "it", "i", "me", "myself", "himself", "herself", "itself"], "Sing"),
    **dict.fromkeys(["we", "they", "us", "them", "ourselves", "themselves"], "Plur"),
}

# POS tag-based number hints
SING_TAGS = {"NN", "NNP"}
PLUR_TAGS = {"NNS", "NNPS"}


def get_number(token: Token) -> list:
    """Determine the 'number' feature for a spacy Token.

    Order of checks:
      1. Pronoun lookup
      2. SpaCy morphological 'Number'
      3. POS tag fallback
      4. Default to 'Sing'.
    """
    lemma = token.lemma_.lower()
    tag = token.tag_
    # 1. Pronoun-based number
    if lemma in PRONOUN_NUMBER:
        return [PRONOUN_NUMBER[lemma]]

    # 2. SpaCy morphological number
    num_feats = token.morph.get("Number", None)
    if num_feats:
        return num_feats
    # 3. POS tag-based fallback
    if tag in SING_TAGS:
        return ["Sing"]
    if tag in PLUR_TAGS:
        return ["Plur"]

    # 4. Default
    return ["Sing"]


def check_agreement(pronoun: Token, candidate: Token) -> tuple[bool, bool]:
    """Check whether a pronoun and a candidate token agree in terms of number and gender.

    Returns a tuple of two booleans. The first boolean indicates whether the agreement check
    succeeded. The second boolean is True if the agreement check passed due to the singular
    'they' case, and False otherwise. This can be used to filter out cases where the agreement
    check is not very informative.

    The agreement check is done in two parts: number and gender.

    For number agreement, the check is done by looking at the morphological features of the
    pronoun and the candidate. If the pronoun is a singular pronoun (e.g. 'he', 'she', 'it'),
    then the candidate must also be singular. If the pronoun is a plural pronoun (e.g. 'they'),
    then the candidate can be either singular or plural. There are two special cases:

    - If the candidate is a person and the pronoun is 'they', then the agreement check passes.
    - If the candidate is a collective noun (e.g. 'team', 'family') and the pronoun is 'they',
      then the agreement check passes.

    For gender agreement, the check is done by looking at the morphological features of the
    pronoun and the candidate. If the pronoun is a gendered pronoun (e.g. 'he', 'she'), then
    the candidate must have the same gender. If the pronoun is a neuter pronoun (e.g. 'it'),
    then the candidate must also be neuter. If the candidate is unspecified, then the
    agreement check passes.

    :param pronoun: The pronoun token.
    :param candidate: The candidate token.
    :return: A tuple of two booleans. The first boolean indicates whether the agreement check
             succeeded. The second boolean is True if the agreement check passed due to the
             singular 'they' case, and False otherwise.
    """
    # --- Number Agreement ---
    pronoun_number = set(get_number(pronoun))
    candidate_number = set(get_number(candidate))
    is_singular_they_case = False
    is_collective_noun_case = False
    # Direct overlap means no mismatch
    if not pronoun_number & candidate_number:
        # Singular 'they' for PERSON
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
            return False, False  # Failed number agreement

    # --- Gender Agreement ---
    pron_gender_set = set(get_gender(pronoun))
    cand_gender_set = set(get_gender(candidate))

    # Case 1: Pronoun is Neuter ('it')
    if "Neut" in pron_gender_set:
        # 'it' should only match explicit Neut. Disallow matching Unspecified, Masc, Fem.
        if "Neut" not in cand_gender_set:
            # print(f"Debug Agreemnt [Gender Fail]: 'it' vs non-Neut {candidate} ({cand_gender_set})")
            return False, False
        # Pass if candidate is Neut
        return True, False  # Agreement OK, not singular they

    # Case 2: Pronoun is Gendered ('he', 'she') or Plural ('they')
    # Check for explicit clashes (Masc vs Fem) only if candidate is NOT Unspecified
    if "Unspecified" not in cand_gender_set and (
        ("Masc" in pron_gender_set and "Fem" in cand_gender_set)
        or ("Fem" in pron_gender_set and "Masc" in cand_gender_set)
    ):
        return False, False

    # If no explicit clash, allow match.
    # - 'he'/'she' can match Masc/Fem respectively, or Unspecified.
    # - 'they' can match Masc, Fem, Neut, or Unspecified (plural or singular).
    return True, is_singular_they_case


# --- Other Helpers (Reflexive, Subject, Sentence, Speaker, Pleonastic - unchanged from v3) ---


def is_reflexive(token: Token) -> bool:
    """Determine whether a token is a reflexive pronoun (e.g., 'himself', 'herself').

    A reflexive pronoun in English ends with 'self' and has the 'PRP' tag.

    :param token: The spaCy token to check.
    :return: True if reflexive pronoun, False otherwise.
    """
    return token.tag_ == "PRP" and token.lemma_.endswith("self")


def find_subject(token: Token):
    head = token.head
    while head.pos_ not in ("VERB", "AUX") and head.dep_ != "ROOT" and head.head != head:
        head = head.head
    if head.pos_ in ("VERB", "AUX") or head.dep_ == "ROOT":
        subjects = [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            subjects = [c for c in head.children if c.dep_ in ("csubj", "csubjpass")]
        if subjects:
            core_subj = subjects[0]
            while core_subj.dep_ == "compound" and core_subj.head.i < token.i:
                core_subj = core_subj.head
            if core_subj.ent_type_ != "PERSON" and core_subj.pos_ != "PROPN":
                person_in_subj = [t for t in core_subj.subtree if t.ent_type_ == "PERSON" and t.i < core_subj.i]
                if person_in_subj:
                    return person_in_subj[-1]
            return core_subj
    return None


def find_speaker(pronoun: Token) -> Token | None:
    current = pronoun
    while current.head != current and current.sent == pronoun.sent:
        governing_verb = current.head
        if governing_verb.lemma_ in REPORTING_VERBS:
            if current.dep_ in ("ccomp", "advcl", "xcomp") or (current.dep_ == "dobj" and current.pos_ == "PRON"):
                speaker = find_subject(governing_verb)
                if speaker:
                    return speaker
                else:
                    return None  # Stop search
        if current.dep_ == "nsubj" and governing_verb.dep_ == "ccomp" and governing_verb.head.lemma_ in REPORTING_VERBS:
            reporting_verb = governing_verb.head
            speaker = find_subject(reporting_verb)
            if speaker:
                return speaker
            else:
                return None
        current = current.head
    return None


def is_pleonastic_it(token: Token) -> bool:
    if token.lemma_ != "it":
        return False
    if token.dep_ == "expl":
        return True
    if token.dep_ == "nsubj":
        verb = token.head
        if verb.pos_ == "VERB":
            if verb.lemma_ in {"rain", "snow", "hail", "thunder", "lighten", "be"}:
                if verb.lemma_ == "be":
                    attr = next((c for c in verb.children if c.dep_ == "attr"), None)
                    if attr and (attr.ent_type_ == "TIME" or any(t.ent_type_ == "TIME" for t in attr.subtree)):
                        return True
                    if attr and any(
                        t.text.lower() in ["o'clock", "pm", "am", "noon", "midnight", "raining", "snowing"]
                        for t in attr.subtree
                    ):
                        return True
                else:
                    return True  # Direct weather verbs
            if verb.lemma_ in {"seem", "appear", "happen", "matter", "turn out", "look", "sound"}:
                if any(c.dep_ in ("ccomp", "csubj", "xcomp", "acomp", "advcl") for c in verb.children):
                    return True
        elif verb.lemma_ == "be" and verb.pos_ == "AUX":
            attr = next((c for c in verb.children if c.dep_ == "attr"), None)
            if attr:
                is_time_attr = (
                    attr.ent_type_ == "TIME"
                    or any(t.ent_type_ == "TIME" for t in attr.subtree)
                    or any(t.text.lower() in ["o'clock", "pm", "am", "noon", "midnight"] for t in attr.subtree)
                )
                has_almost = any(c.lemma_ == "almost" and c.dep_ == "advmod" for c in attr.children)
                is_num_like = attr.like_num or (len(attr.text) > 0 and attr.text[0].isdigit())
                if is_time_attr or (has_almost and is_num_like):
                    return True
                relcl = next((c for c in verb.children if c.dep_ == "relcl" and c.head == attr), None)
                if relcl and relcl[0].tag_ in ["WP", "WDT"]:
                    return True  # Cleft
    return False


# --- End Helper Functions ---


# --- Main Resolution Function (using the updated helpers) ---
# The main function rule_based_coref_resolution_v3 remains unchanged in its
# structure and rule logic, as the changes were within the helper functions.
# Make sure to call the function with the updated helpers integrated.


def rule_based_coref_resolution_v4(  # Renamed function
    text: str, similarity_threshold: float = 0.5, use_similarity_fallback: bool = False, search_sentences: int = 2
):
    """Applies v4 rule-based coreference resolution with Unspecified Gender handling."""
    doc = nlp(text)
    coref_results_internal = []
    processed_mentions = set()

    # --- Processing Loop (Identical structure to v3) ---
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

            # --- A. Pronoun Resolution ---
            is_personal_pronoun = token.tag_ == "PRP"
            is_possessive_pronoun = token.tag_ == "PRP$"
            is_relative_possessive = token.tag_ == "WP$"
            is_relative_nonpossessive = token.tag_ in ["WP", "WDT"]
            is_reflexive_pronoun = is_reflexive(token)

            if is_personal_pronoun or is_possessive_pronoun or is_relative_possessive or is_relative_nonpossessive:
                # Rule 0: Skip Pleonastic 'It'
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

                # Rule 2a: Relative Non-Possessive ('who', 'which', 'that')
                elif is_relative_nonpossessive:
                    potential_antecedent = token.head
                    if potential_antecedent.pos_ in ("ADP", "AUX", "VERB"):
                        potential_antecedent = potential_antecedent.head
                    if potential_antecedent.pos_ in {"NOUN", "PROPN", "PRON"}:
                        agrees, _ = check_agreement(token, potential_antecedent)  # Uses NEW check_agreement
                        if agrees:
                            antecedent = potential_antecedent
                            confidence = 0.92
                            rule = "Relative Pronoun -> Syntactic Head"

                # Rule 3: Quoted Speech Pronouns ('I', 'me', 'my', 'we', 'us', 'our')
                if not antecedent and token.lemma_ in {"i", "me", "my", "we", "us", "our"}:
                    speaker = find_speaker(token)  # Uses NEW find_speaker
                    if speaker:
                        agrees, _ = check_agreement(token, speaker)  # Uses NEW check_agreement
                        if agrees:
                            antecedent = speaker
                            confidence = 0.90
                            rule = "Quoted Pronoun -> Speaker"

                # Rule 2b / 4: Possessive / Relative 'whose'
                elif not antecedent and (is_possessive_pronoun or is_relative_possessive):
                    potential_candidates = []
                    search_rule_name = (
                        "Possessive Antecedent" if is_possessive_pronoun else "Relative Possessive (whose) Antecedent"
                    )
                    for j in range(token.i - 1, start_search_token_idx - 1, -1):  # Backward search loop
                        if j < 0:
                            break
                        candidate = doc[j]
                        if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                            continue
                        agrees, _ = check_agreement(token, candidate)  # Uses NEW check_agreement
                        if not agrees:
                            continue
                        # Scoring (unchanged from v3 except uses new check_agreement implicitly)
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
                        if candidate.lemma_ in PERSONAL_PRONOUN_LEMMAS:
                            cand_score *= 0.70
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
                    potential_candidates = []
                    for j in range(token.i - 1, start_search_token_idx - 1, -1):  # Backward search loop
                        if j < 0:
                            break
                        candidate = doc[j]
                        if candidate.pos_ not in {"NOUN", "PROPN", "PRON"} or is_reflexive(candidate):
                            continue
                        agrees, is_singular_they = check_agreement(token, candidate)  # Uses NEW check_agreement
                        if not agrees:
                            continue
                        # Scoring (unchanged from v3 except uses new check_agreement implicitly)
                        cand_score = 0.15
                        cand_rule_detail = "Agreement"
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
                            cand_rule_detail += " (Subject)" if cand_rule_detail != "Agreement" else "Subject Salience"
                        if candidate.lemma_ in PERSONAL_PRONOUN_LEMMAS:
                            cand_score *= 0.70
                        distance = token.i - candidate.i
                        proximity_factor = max(0.1, 1.0 - (distance / 75.0))
                        cand_score *= proximity_factor
                        cand_score = min(cand_score, 1.0)
                        if cand_rule_detail == "Agreement":
                            cand_rule_detail += " + Proximity"
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

                # Set antecedent from backward search if found & no specific rule applied
                if best_candidate_info and not antecedent:
                    antecedent = best_candidate_info["token"]
                    confidence = best_candidate_info["score"]
                    rule = best_candidate_info["reason"]

            # --- B. Proper Noun (PN) Coreference ---
            elif token.pos_ == "PROPN":
                potential_pn_antecedents = []
                for j in range(token.i - 1, start_search_token_idx - 1, -1):  # Backward search loop
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
                    best_pn_match_info = potential_pn_antecedents[0]
                    if best_pn_match_info["type"] == "Exact":  # Check for override
                        for cand_info in potential_pn_antecedents:
                            if (
                                cand_info["type"] == "Partial"
                                and best_pn_match_info["token"].text in cand_info["token"].text.split()
                            ):
                                best_pn_match_info = cand_info
                                break
                    best_pn_token = best_pn_match_info["token"]
                    if best_pn_token.i != token.i and best_pn_token.i not in processed_mentions:
                        antecedent = best_pn_token
                        confidence = best_pn_match_info["score"]
                        rule = best_pn_match_info["rule"]

            # --- Store Result ---
            if antecedent and antecedent.i != token.i:
                coref_results_internal.append((token, antecedent, round(confidence, 2), rule))
                processed_mentions.add(token.i)

    # --- Convert to Final Output Format ---
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
# [Include the same testing samples and loop as before]
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
    "Cleft It": "It was Susan who solved the puzzle.",  # It -> Pleonastic, who -> Susan
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


print("--- Running Coreference Resolution v4 (Unspecified Gender) ---")
for description, text in all_samples.items():
    print(f"\n--- [{description}] ---")
    print(f"Text: {text}")
    try:
        pairs_with_indices = rule_based_coref_resolution_v4(
            text, search_sentences=2, use_similarity_fallback=False
        )  # Call v4 function
        print(f"Coreference pairs (Mention Span, Antecedent Span, Confidence, Rule):")
        if pairs_with_indices:
            for pair in pairs_with_indices:
                mention_span, antecedent_span, conf, rule = pair
                print(
                    f"  - Mention: '{mention_span['text']}' ({mention_span['start']}:{mention_span['end']}) -> "
                    f"Antecedent: '{antecedent_span['text']}' ({antecedent_span['start']}:{antecedent_span['end']}) "
                    f"(Conf: {conf:.2f}, Rule: {rule})"
                )
        else:
            print("  No pairs found.")

    except Exception as e:
        print(f"\n!!! An error occurred processing '{description}': {e} !!!")
        import traceback

        traceback.print_exc()
