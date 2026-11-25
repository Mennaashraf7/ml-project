import re
import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))


def clean_text(text):
    if not isinstance(text, str):
        return ""

    #removing html tags
    text = re.sub(r"<[^>]*>", " ", text)

    #lowercasing
    text = text.lower()

    # replace everything that is not a letter (punctuation, numbers, symbols) with a space
    text = re.sub(r"[^a-z]", " ", text)

    #split into words (also collapses repeated spaces)
    words = text.split()

    #drop stopwords and single-letter leftovers
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]

    #join back into one string
    return " ".join(words)