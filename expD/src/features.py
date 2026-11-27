"""Feature extractors for Paper D. CPU only.
- TF-IDF (char+word) for the cheap baseline.
- multilingual-e5-small sentence embeddings for the cross-lingual detector.
Embeddings are cached to disk (npy) keyed by a hash of the text list.
"""
import os, hashlib, numpy as np

_E5 = None  # lazy singleton


def tfidf_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    # word + char n-grams catch both lexical injection cues and obfuscation
    return TfidfVectorizer(
        ngram_range=(1, 2), analyzer="word", min_df=2, max_features=40000,
        sublinear_tf=True, lowercase=True,
    )


def _load_e5():
    global _E5
    if _E5 is None:
        from sentence_transformers import SentenceTransformer
        # small (~118M), multilingual, good on CPU
        _E5 = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
    return _E5


def e5_embed(texts, cache_dir="expD/data/emb_cache", batch_size=32, prefix="query: "):
    """Return (N, D) float32 embeddings. e5 expects a 'query: ' / 'passage: ' prefix."""
    os.makedirs(cache_dir, exist_ok=True)
    key = hashlib.md5((prefix + "||".join(texts)).encode("utf-8")).hexdigest()
    path = os.path.join(cache_dir, key + ".npy")
    if os.path.exists(path):
        return np.load(path)
    model = _load_e5()
    emb = model.encode([prefix + t for t in texts], batch_size=batch_size,
                       convert_to_numpy=True, normalize_embeddings=True,
                       show_progress_bar=True)
    np.save(path, emb.astype(np.float32))
    return emb.astype(np.float32)
