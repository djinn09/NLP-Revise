def summary_nltk(text: str, top_n_sentences: int = 3, custom_stop_words: set = None) -> str:
    """
    Generates a summary of the input text using NLTK.

    Args:
        text: The input string to summarize.
        top_n_sentences: The number of sentences to include in the summary.
        custom_stop_words: An optional set of stop words. If None, uses nltk_stop_words.

    Returns:
        A string containing the summary.
    """
    # 'nltk_stop_words' and 'custom_punctuation' should be available globally or passed in.
    # 'word_tokenize', 'sent_tokenize' from nltk.tokenize should be imported.
    # 'heapq.nlargest' should be imported.

    if custom_stop_words is None:
        # Use the globally defined nltk_stop_words if no specific set is provided
        stop_words_to_use = nltk_stop_words 
    else:
        stop_words_to_use = custom_stop_words

    # 1. Tokenize words and convert to lower case for consistent counting
    words = word_tokenize(text.lower())
    
    # 2. Calculate word frequencies
    word_frequencies = {}
    for word in words:
        # Filter out stop words and punctuation.
        # .isalnum() helps to remove tokens that are just punctuation or mixed non-alphanumeric.
        if word not in stop_words_to_use and word not in custom_punctuation and word.isalnum():
            if word not in word_frequencies: # No .keys() needed here
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    if not word_frequencies:
        return "" # Return empty string if no relevant words found

    maximum_frequency = max(word_frequencies.values())
    
    # 3. Normalize word frequencies
    for word in word_frequencies: # No .keys() needed here
        word_frequencies[word] = word_frequencies[word] / maximum_frequency
        
    # 4. Tokenize sentences
    sentence_list = sent_tokenize(text)
    sentence_scores = {}
    
    # 5. Score sentences
    for sent in sentence_list:
        # Tokenize words in the current sentence for scoring
        # Using word_tokenize for consistency with how word frequencies were calculated
        current_sentence_words = word_tokenize(sent.lower())
        for word in current_sentence_words:
            if word in word_frequencies: # No .keys() needed here
                # This condition (sentence length < 30 words) is from the original notebook.
                # It might be useful to make this limit a parameter or add a comment explaining its purpose.
                if len(sent.split(' ')) < 30: 
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    
    if not sentence_scores:
        return "" # Return empty if no sentences could be scored

    # 6. Select top N sentences
    # Ensure that top_n_sentences does not exceed the number of available scored sentences
    actual_top_n = min(top_n_sentences, len(sentence_scores))
    summary_sentences_selected = heapq.nlargest(actual_top_n, sentence_scores, key=sentence_scores.get)
    
    # 7. Join sentences to form the summary
    # Ensure each sentence is stripped of leading/trailing whitespace and newlines are removed.
    summary = ' '.join(s.strip().replace("\n", "") for s in summary_sentences_selected)
    return summary
