"""TF-IDF feature extraction for the classical baseline.

Word n-grams capture phrases such as "claim your prize" or "kyc update karo".
Character n-grams (char_wb, 2-5) are robust to the spelling variation that is common
in SMS and in Romanized Hindi ("nahi"/"nhi"/"nahin", "karo"/"kro", "aapka"/"apka").
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.preprocessing import clean_text

# Token pattern keeps placeholder tokens such as <url> and <phone> intact.
TOKEN_PATTERN = r"<\w+>|[\w£$€₹%']+"


def build_vectorizer(word_ngrams=(1, 2), char_ngrams=(2, 5), min_df=2, max_features_word=30000,
                     max_features_char=60000) -> FeatureUnion:
    word = TfidfVectorizer(preprocessor=clean_text, token_pattern=TOKEN_PATTERN, ngram_range=word_ngrams,
                           min_df=min_df, max_features=max_features_word, sublinear_tf=True, lowercase=False)
    char = TfidfVectorizer(preprocessor=clean_text, analyzer="char_wb", ngram_range=char_ngrams,
                           min_df=min_df, max_features=max_features_char, sublinear_tf=True, lowercase=False)
    return FeatureUnion([("word", word), ("char", char)])
