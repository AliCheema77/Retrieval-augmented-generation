import numpy as np
from pathlib import Path

from chunker import load_text, chunk_text
from embedder import model, embed_chunks
from bm25 import build_bm25_index, bm25_search

SCRIPT_DIR = Path(__file__).parent


def cosine_similarity(a, b):
    """Measure how semantically similar two embedding vectors are (1 = same direction/meaning, -1 = opposite)."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def retrieve(query: str, chunks: list[str], chunk_embeddings, top_k: int = 3):
    """Pure vector (semantic) search: rank chunks by cosine similarity to the query and return the top-k."""
    query_embedding = model.encode(query)

    scores = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    top_chunks = []
    for score, i in scores[:top_k]:
        top_chunks.append((chunks[i], score))

    return top_chunks


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge multiple ranked lists (e.g. vector search + BM25) into one ranking by each item's position, not its raw score — avoids having to normalize incompatible score scales."""
    fused_scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(query: str, chunks: list[str], chunk_embeddings, bm25_index: dict, top_k: int = 3):
    """Combine semantic (vector) and keyword (BM25) search results via reciprocal rank fusion for retrieval that catches both meaning and exact terms."""
    query_embedding = model.encode(query)
    vector_scores = [(i, cosine_similarity(query_embedding, e)) for i, e in enumerate(chunk_embeddings)]
    vector_scores.sort(key=lambda x: x[1], reverse=True)
    vector_ranking = [i for i, _ in vector_scores]

    bm25_ranking = [i for i, _ in bm25_search(query, bm25_index)]

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

    return [(chunks[doc_id], score) for doc_id, score in fused[:top_k]]


if __name__ == "__main__":
    sample_path = SCRIPT_DIR / "sample.txt"
    text = load_text(sample_path)
    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)

    query = "What are backend skills?"
    bm25_index = build_bm25_index(chunks)
    results = hybrid_retrieve(query, chunks, chunk_embeddings, bm25_index)

    for chunk, score in results:
        print(f"Score: {score:.4f}")
        print(f"Chunk: {chunk}\n")