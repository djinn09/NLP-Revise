def summary_spacy(text: str, percentage: float = 0.4) -> str:
    """
    Generates a summary of the input text using Spacy.

    Args:
        text: The input string to summarize.
        percentage: The desired percentage of the original text length for the summary.
                    (e.g., 0.4 for 40% summary).

    Returns:
        A string containing the summary.
    """
    doc = nlp(text) # 'nlp' should be pre-loaded (e.g., spacy.load("en_core_web_sm"))
                    # 'spacy_stop_words' and 'custom_punctuation' should be available globally or passed in.

    # 1. Calculate word frequencies
    word_frequencies = Counter(
        word.text.lower() for word in doc
        if word.text.lower() not in spacy_stop_words  # Using pre-defined spacy_stop_words
        and word.text.lower() not in custom_punctuation # Using pre-defined custom_punctuation
        and not word.is_punct
        and not word.is_space
    )

    if not word_frequencies:
        return ""  # Return empty string if no relevant words found

    max_frequency = max(word_frequencies.values())

    # 2. Normalize word frequencies
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # 3. Score sentences
    sentence_list = [sent for sent in doc.sents]
    sentence_scores = {}
    for sent in sentence_list:
        for word in sent:
            if word.text.lower() in word_frequencies:
                if sent not in sentence_scores:
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]
    
    if not sentence_scores:
        return "" # Return empty if no sentences could be scored

    # 4. Determine number of sentences for summary
    # Ensure at least one sentence if content and scores exist, and percentage is very small.
    num_summary_sentences = int(len(sentence_scores) * percentage)
    if num_summary_sentences < 1 and len(sentence_scores) > 0:
        num_summary_sentences = 1
    
    # 5. Select top N sentences
    summary_sentence_objects = heapq.nlargest(
        num_summary_sentences,
        sentence_scores,
        key=sentence_scores.get
    )

    # 6. Join sentences to form the summary
    summary = "".join(sent.text.strip().replace("\n", "") for sent in summary_sentence_objects)
    return summary
