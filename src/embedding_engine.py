"""
Embedding Engine — Sentence Transformers (BERT) for semantic text encoding.
Uses all-MiniLM-L6-v2 for efficient 384-dim embeddings.
Includes TF-IDF fallback for robustness on constrained environments.
"""

import os
import numpy as np
from typing import List, Union, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class EmbeddingEngine:
    """
    Production embedding engine with Sentence Transformers (primary)
    and TF-IDF (fallback). Designed for Streamlit Cloud deployment.
    """

    def __init__(self, use_transformers: bool = True):
        self.use_transformers = use_transformers
        self._model = None
        self._tfidf = None
        self._tfidf_fitted = False

    def _load_transformer_model(self):
        """Lazy-load the Sentence Transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as e:
                print(f"⚠ Transformer model load failed: {e}. Falling back to TF-IDF.")
                self.use_transformers = False
        return self._model

    def encode(self, texts: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Encode text(s) into dense vector embeddings.

        Args:
            texts: Single string or list of strings to encode.
            show_progress: Show progress bar during encoding.

        Returns:
            numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Filter empty strings
        texts = [t if t and t.strip() else "empty document" for t in texts]

        if self.use_transformers:
            try:
                model = self._load_transformer_model()
                if model is not None:
                    embeddings = model.encode(
                        texts,
                        show_progress_bar=show_progress,
                        batch_size=32,
                        normalize_embeddings=True,
                    )
                    return np.array(embeddings)
            except Exception as e:
                print(f"⚠ Encoding failed: {e}. Falling back to TF-IDF.")

        # Fallback: TF-IDF
        return self._tfidf_encode(texts)

    def _tfidf_encode(self, texts: List[str]) -> np.ndarray:
        """TF-IDF fallback encoding."""
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(
                max_features=EMBEDDING_DIMENSION,
                stop_words='english',
                ngram_range=(1, 2),
                sublinear_tf=True,
            )

        if not self._tfidf_fitted:
            vectors = self._tfidf.fit_transform(texts).toarray()
            self._tfidf_fitted = True
        else:
            vectors = self._tfidf.transform(texts).toarray()

        # Pad or truncate to match embedding dimension
        if vectors.shape[1] < EMBEDDING_DIMENSION:
            padding = np.zeros((vectors.shape[0], EMBEDDING_DIMENSION - vectors.shape[1]))
            vectors = np.hstack([vectors, padding])
        elif vectors.shape[1] > EMBEDDING_DIMENSION:
            vectors = vectors[:, :EMBEDDING_DIMENSION]

        # Normalize
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vectors = vectors / norms

        return vectors

    def compute_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        if embedding_a.ndim == 1:
            embedding_a = embedding_a.reshape(1, -1)
        if embedding_b.ndim == 1:
            embedding_b = embedding_b.reshape(1, -1)
        return float(cosine_similarity(embedding_a, embedding_b)[0][0])

    def compute_similarity_batch(self, query_embedding: np.ndarray, corpus_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between a query and a corpus of embeddings."""
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, corpus_embeddings)
        return similarities.flatten()

    def save_embeddings(self, embeddings: np.ndarray, path: str):
        """Save precomputed embeddings to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, embeddings)

    def load_embeddings(self, path: str) -> Optional[np.ndarray]:
        """Load precomputed embeddings from disk."""
        if os.path.exists(path):
            return np.load(path)
        return None
