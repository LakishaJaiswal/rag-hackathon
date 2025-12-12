# search.py
import argparse
import pickle
import os
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from config import INDEX_DIR
from sentence_transformers import SentenceTransformer

# ---- Load data ----
corpus = pickle.load(open(os.path.join(INDEX_DIR, "corpus.pkl"), "rb"))
metadata = pickle.load(open(os.path.join(INDEX_DIR, "meta.pkl"), "rb"))

# BM25
bm25 = pickle.load(open(os.path.join(INDEX_DIR, "bm25.pkl"), "rb"))

# Dense embeddings + FAISS
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

# ---- Retrieval Methods ----
def search_bm25(query, top_k=5):
    scores = bm25.get_scores(query.split())
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(i, scores[i]) for i in top_idx]

def search_dense(query, top_k=5):
    q_emb = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(q_emb, dtype=np.float32), top_k)
    return [(int(I[0][i]), float(D[0][i])) for i in range(top_k)]

def search_hybrid(query, top_k=5, alpha=0.5):
    # BM25
    bm25_scores = bm25.get_scores(query.split())
    bm25_norm = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + 1e-9)

    # Dense
    q_emb = embedding_model.encode([query])
    D, I = faiss_index.search(np.array(q_emb, dtype=np.float32), len(corpus))
    dense_scores = np.zeros(len(corpus))
    for i, d in zip(I[0], D[0]):
        dense_scores[i] = d
    dense_norm = (dense_scores - np.min(dense_scores)) / (np.max(dense_scores) - np.min(dense_scores) + 1e-9)

    # Combine
    scores = alpha * bm25_norm + (1 - alpha) * dense_norm
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(i, scores[i]) for i in top_idx]

# ---- Pretty Print ----
def display_results(results, query):
    print("\n🔎 Query:", query)
    print("="*60)
    citations = []
    for rank, (idx, score) in enumerate(results, 1):
        doc_id = metadata[idx]["doc_id"]
        chunk_id = metadata[idx]["chunk_id"]
        snippet = metadata[idx]["text"][:200].replace("\n", " ")
        print(f"[{rank}] (Doc {doc_id} | Chunk {chunk_id} | Score={score:.4f})")
        print(f"📝 {snippet}...\n")
        citations.append(f"Doc{doc_id}_Chunk{chunk_id}")
    print("📖 Sources:", ", ".join(citations))

# ---- Main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Query text")
    parser.add_argument("--method", type=str, default="hybrid", help="bm25 | dense | hybrid")
    parser.add_argument("--top_k", type=int, default=5, help="Top K results")
    parser.add_argument("--alpha", type=float, default=0.6, help="Weight for BM25 in hybrid search")
    args = parser.parse_args()

    if args.method == "bm25":
        results = search_bm25(args.query, args.top_k)
    elif args.method == "dense":
        results = search_dense(args.query, args.top_k)
    else:
        results = search_hybrid(args.query, args.top_k, args.alpha)

    display_results(results, args.query)
