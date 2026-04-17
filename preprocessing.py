import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


@lru_cache(maxsize=1)
def get_text_tools():
    """Load NLTK resources once and reuse them across training/prediction."""
    try:
        words = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = stopwords.words("english")

    return set(words), PorterStemmer()


def clean_text(text):
    """Normalize email text for the Naive Bayes model."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    stop_words, stemmer = get_text_tools()
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)
