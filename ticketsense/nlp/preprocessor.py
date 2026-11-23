"""Text preprocessing and cleaning pipeline for TicketSense-AI."""

import re
from typing import List, Union, Any
from sklearn.base import BaseEstimator, TransformerMixin


# Common English contractions mapping
CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bi'm\b": "i am",
    r"\bi've\b": "i have",
    r"\bi'll\b": "i will",
    r"\bi'd\b": "i would",
    r"\byou're\b": "you are",
    r"\byou've\b": "you have",
    r"\byou'll\b": "you will",
    r"\bit's\b": "it is",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwouldn't\b": "would not",
    r"\bshouldn't\b": "should not",
    r"\bcouldn't\b": "could not",
}


def clean_text(text: str) -> str:
    """Clean and normalize a customer support message.

    Steps:
        1. Convert to lowercase.
        2. Replace URLs and email addresses.
        3. Expand common contractions.
        4. Normalize punctuation and remove special symbols.
        5. Collapse multiple whitespaces.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # Expand contractions
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    # Replace special characters/symbols with space, keep letters, numbers, and basic apostrophes
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text


class TextCleaner(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for text cleaning."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[List[str], Any]) -> List[str]:
        if hasattr(X, "tolist"):
            items = X.tolist()
        else:
            items = list(X)
        return [clean_text(item) for item in items]
