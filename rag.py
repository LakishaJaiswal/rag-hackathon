import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from config import INDEX_DIR, MODEL_PATH
from llama_cpp import Llama
import json

# Paths
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")
BM25_PATH = os.path.join(INDEX_DIR, "bm25.pkl")

# ---- Load or Build Index & Metadata ----
def load_index_and_meta(knowledge_file=None):
    """
    Load FAISS index, metadata, and BM25 object from disk.
    If missing and knowledge_file is provided, build them.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH) or not os.path.exists(BM25_PATH):
        if knowledge_file is None:
            raise FileNotFoundError("Indexes not found. Provide knowledge_file to build index.")

        # Load knowledge JSON
        with open(knowledge_file, "r", encoding="utf-8") as f:
            knowledge = json.load(f)

        # Use 'Snippet' as the text field
        knowledge = [doc for doc in knowledge if "Snippet" in doc]
        if not knowledge:
            raise ValueError("No valid documents with 'Snippet' found in knowledge file.")

        texts = [doc["Snippet"] for doc in knowledge]
        meta = knowledge

        print(f"✅ Building FAISS index and BM25 with {len(meta)} documents...")

        # Build FAISS index
        embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")  # better retrieval quality
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(index, INDEX_PATH)

        # Build BM25
        tokenized_texts = [t.split() for t in texts]
        bm25 = BM25Okapi(tokenized_texts)
        with open(META_PATH, "wb") as f:
            pickle.dump(meta, f)
        with open(BM25_PATH, "wb") as f:
            pickle.dump(bm25, f)

        print("✅ Index building complete.")

    else:
        # Load from disk
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        with open(BM25_PATH, "rb") as f:
            bm25 = pickle.load(f)
        print(f"✅ Loaded existing FAISS index with {len(meta)} documents")

    return index, meta, bm25

# ---- Hybrid Retrieval ----
def retrieve(query, index, meta, bm25, top_k=10, alpha=0.6):
    """
    Retrieve top_k documents using hybrid BM25 + dense embeddings.
    Returns list of dicts with doc_id, chunk_id, snippet, score, title, url.
    """
    # BM25 scores
    bm25_scores = bm25.get_scores(query.split())
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)

    # Dense embeddings
    embedder = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb, dtype=np.float32), len(meta))
    dense_scores = np.zeros(len(meta))
    for i, d in zip(I[0], D[0]):
        dense_scores[i] = d
    dense_norm = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-9)

    # Combine scores
    scores = alpha * bm25_norm + (1 - alpha) * dense_norm
    top_idx = np.argsort(scores)[::-1][:top_k]

    # Build results
    results = []
    for idx in top_idx:
        results.append({
            "doc_id": meta[idx]["chunk_id"],  # use chunk_id as identifier
            "chunk_id": 0,
            "snippet": meta[idx]["Snippet"],
            "title": meta[idx].get("Title", ""),
            "url": meta[idx].get("URL", ""),
            "score": float(scores[idx])
        })

    return results
