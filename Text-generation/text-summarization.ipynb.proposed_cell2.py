# Load spacy model
# Ensure you have the model downloaded, e.g., by running:
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Define custom punctuation by extending string.punctuation
# Added common quote characters and em-dash often found in texts
# Note: string_punctuation was imported as 'string_punctuation' to avoid conflict
# with the notebook's original 'punctuation' variable if it were to be kept.
# However, it's better to use this custom_punctuation consistently.
custom_punctuation = string_punctuation + "\n" + "“" + "”" + "–" 

# NLTK Stop words (can be passed to functions or defined locally for better modularity)
# These are used by summary_nltk and preprocess_data (for LSA)
nltk_stop_words = set(stopwords.words("english"))

# Spacy Stop words (can be accessed via nlp.Defaults.stop_words or imported)
# These are used by summary_spacy
spacy_stop_words = nlp.Defaults.stop_words # or from spacy.lang.en.stop_words import STOP_WORDS
# It's also possible to extend this set if needed:
# spacy_stop_words.update(["some", "additional", "words"])

# The notebook originally had:
# punctuation = punctuation+"\n"+"“" + '”' +"–" 
# stopWords = set(stopwords.words("english"))
# It's better to use the new names 'custom_punctuation' and 'nltk_stop_words'
# to avoid modifying imported modules directly and for clarity.
