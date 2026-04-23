"""
Text Processor — NLTK-based text preprocessing pipeline for resumes and job descriptions.
Handles cleaning, tokenization, lemmatization, and stopword removal.
"""

import re
import string
import nltk

# Download required NLTK data (silent)
for resource in ['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}' if resource in ['stopwords', 'wordnet'] else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Custom Stopwords for HR/Resume Domain ─────────────────────────────────
HR_STOPWORDS = {
    "resume", "curriculum", "vitae", "cv", "page", "phone", "email",
    "address", "date", "birth", "nationality", "gender", "marital",
    "status", "references", "available", "upon", "request", "objective",
    "summary", "profile", "dear", "sir", "madam", "sincerely",
    "regards", "thank", "position", "apply", "applying", "application",
    "company", "organization", "responsible", "responsibilities",
    "duties", "role", "worked", "working", "work", "job", "employed",
    "employment", "employer", "candidate", "applicant",
}

class TextProcessor:
    """Production-grade text preprocessing engine for resume analysis."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')).union(HR_STOPWORDS)

    def clean_text(self, text: str) -> str:
        """Deep clean raw text from resume/JD."""
        if not text or not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', ' ', text)
        # Remove phone numbers
        text = re.sub(r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\./0-9]{7,15}', ' ', text)
        # Remove special characters but keep hyphens in compound words
        text = re.sub(r'[^a-zA-Z0-9\s\-\+\#\.]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> list:
        """Tokenize text into words."""
        try:
            return word_tokenize(text.lower())
        except Exception:
            return text.lower().split()

    def lemmatize_tokens(self, tokens: list) -> list:
        """Lemmatize a list of tokens."""
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def remove_stopwords(self, tokens: list) -> list:
        """Remove English + HR-specific stopwords."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]

    def process(self, text: str, keep_original_case: bool = False) -> str:
        """Full preprocessing pipeline: clean → tokenize → lemmatize → rejoin."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        tokens = [t for t in tokens if t not in string.punctuation]
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        return ' '.join(tokens)

    def extract_sentences(self, text: str) -> list:
        """Split text into sentences."""
        try:
            return sent_tokenize(text)
        except Exception:
            return [s.strip() for s in text.split('.') if s.strip()]

    def extract_ngrams(self, text: str, n: int = 2) -> list:
        """Extract n-grams from text."""
        tokens = self.tokenize(self.clean_text(text))
        tokens = self.remove_stopwords(tokens)
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def get_word_frequencies(self, text: str) -> dict:
        """Get word frequency distribution."""
        tokens = self.tokenize(self.clean_text(text))
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize_tokens(tokens)
        freq = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1
        return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
