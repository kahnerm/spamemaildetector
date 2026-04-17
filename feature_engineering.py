import re
from functools import lru_cache

import nltk
import numpy as np
from nltk.corpus import stopwords, words
from scipy.sparse import csr_matrix, hstack


TOKEN_PATTERN = re.compile(r"[a-zA-Z]{3,}")
URL_OR_EMAIL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\S+@\S+")

COMMON_SCAM_MISSPELLINGS = {
    "acount",
    "adress",
    "aproved",
    "authetication",
    "bankingg",
    "benifit",
    "cliam",
    "clickk",
    "confimation",
    "confirmm",
    "delivary",
    "delivry",
    "detials",
    "documment",
    "eligable",
    "imediately",
    "immediatly",
    "informaton",
    "loggin",
    "notifcation",
    "pakage",
    "pasword",
    "paymant",
    "recieve",
    "recieved",
    "restricton",
    "securty",
    "shiping",
    "suspention",
    "suspnded",
    "transcation",
    "urgentt",
    "verfication",
    "verfy",
    "wierd",
}

PROJECT_WORDS = {
    "api",
    "app",
    "bank",
    "billing",
    "cash",
    "csv",
    "dataset",
    "delivery",
    "email",
    "giftcard",
    "login",
    "multinomial",
    "naive",
    "online",
    "password",
    "phishing",
    "profile",
    "scikit",
    "signin",
    "streamlit",
    "subscription",
    "url",
    "username",
    "verification",
    "verify",
    "website",
}


@lru_cache(maxsize=1)
def get_dictionary_words():
    try:
        dictionary = words.words()
        stop_words = stopwords.words("english")
    except LookupError:
        nltk.download("words", quiet=True)
        nltk.download("stopwords", quiet=True)
        dictionary = words.words()
        stop_words = stopwords.words("english")

    return {word.lower() for word in dictionary} | set(stop_words) | PROJECT_WORDS


def is_known_word(token, dictionary):
    if token in dictionary:
        return True

    suffix_forms = []
    if token.endswith("s"):
        suffix_forms.append(token[:-1])
    if token.endswith("es"):
        suffix_forms.append(token[:-2])
    if token.endswith("ed"):
        suffix_forms.extend([token[:-1], token[:-2]])
    if token.endswith("ing"):
        suffix_forms.extend([token[:-3], token[:-3] + "e"])

    return any(form in dictionary for form in suffix_forms if len(form) >= 3)


def spelling_summary(text):
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    dictionary = get_dictionary_words()
    text_without_links = URL_OR_EMAIL_PATTERN.sub(" ", text)
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(text_without_links)]
    misspelled = [
        token
        for token in tokens
        if not is_known_word(token, dictionary)
        and token not in COMMON_SCAM_MISSPELLINGS
    ]
    scam_typos = [token for token in tokens if token in COMMON_SCAM_MISSPELLINGS]

    total_words = len(tokens)
    misspelling_count = len(misspelled) + len(scam_typos)
    misspelling_ratio = misspelling_count / total_words if total_words else 0.0

    return {
        "total_words": total_words,
        "misspelling_count": misspelling_count,
        "misspelling_ratio": misspelling_ratio,
        "possible_misspellings": sorted(set(misspelled + scam_typos)),
        "scam_typo_count": len(scam_typos),
    }


def spelling_feature_matrix(texts):
    rows = []
    for text in texts:
        summary = spelling_summary(text)
        rows.append(
            [
                min(summary["misspelling_count"], 20),
                round(summary["misspelling_ratio"] * 10, 4),
                summary["scam_typo_count"] * 3,
            ]
        )

    return csr_matrix(np.array(rows, dtype=float))


def combine_text_and_spelling_features(vectorizer, raw_texts, cleaned_texts):
    text_features = vectorizer.transform(cleaned_texts)
    spelling_features = spelling_feature_matrix(raw_texts)
    return hstack([text_features, spelling_features], format="csr")
