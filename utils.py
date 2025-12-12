import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load embedding model once globally
_embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed_text(texts):
    """
    Convert text(s) into dense embeddings using SentenceTransformer.
    Supports both single string and list of strings.
    Returns float32 numpy array (FAISS compatible).
    """
    if isinstance(texts, str):
        texts = [texts]
    embeddings = _embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return np.array(embeddings).astype("float32")


def load_json(path):
    """
    Load a JSON file and return as Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    """
    Save Python object as JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors or arrays.
    """
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)
    dot = np.dot(vec1, vec2.T)
    norm = np.linalg.norm(vec1, axis=1)[:, None] * np.linalg.norm(vec2, axis=1)
    return dot / norm
