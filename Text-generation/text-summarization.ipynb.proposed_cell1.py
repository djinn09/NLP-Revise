import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import PorterStemmer

import spacy
# from spacy.lang.en.stop_words import STOP_WORDS # This is available if spacy is loaded.

import heapq
from collections import Counter
from string import punctuation as string_punctuation # Renamed to avoid conflict
from typing import List # For type hinting

from textblob import TextBlob

from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel

# It's good practice to download NLTK resources explicitly if not done elsewhere
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download('punkt', quiet=True)
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download('stopwords', quiet=True)
