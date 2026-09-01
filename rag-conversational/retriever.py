import numpy as np
from pathlib import Path

from chunker import load_text, chunk_text
from embedder import model, embed_chunks
from bm25 import build_bm25_index, bm25_search
from vectorstore import vector_search, ensure_indexed

SCRIPT_DIR = Path(__file__).parent


def cosine_similarity(a, b):
    """
    Measure how semantically similar two embedding vectors are.

    Cosine similarity looks at the *angle* between two vectors rather than
    their raw magnitude, which is what makes it suitable for comparing
    embeddings: two chunks about the same topic should point in roughly
    the same direction in vector space even if their exact magnitudes
    differ.

    Parameters:
        a, b: two embedding vectors (NumPy arrays) of the same dimension.

    Returns:
        A float between -1 and 1: 1 means identical direction/meaning,
        0 means unrelated, -1 means opposite meaning.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Dividing the dot product by both magnitudes cancels out vector length,
    # leaving only the directional (angular) similarity.
    return dot_product / (norm_a * norm_b)


def retrieve(query: str, chunks: list[str], chunk_embeddings, top_k: int = 3):
    """
    Pure vector (semantic) search: find the chunks whose meaning is closest
    to the query's meaning, ignoring exact keyword overlap.

    Parameters:
        query: the raw question text.
        chunks: the original text chunks (parallel list to chunk_embeddings).
        chunk_embeddings: precomputed embedding vector for each chunk.
        top_k: how many top-ranked chunks to return.

    Returns:
        A list of (chunk_text, similarity_score) tuples, best match first.
    """
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
    """
    Merge multiple independently-ranked lists (e.g. vector search results
    and BM25 results) into a single combined ranking.

    Reciprocal Rank Fusion (RRF) combines results using each item's
    *position* in each ranking rather than its raw score. This avoids the
    problem of vector similarity scores (roughly 0-1) and BM25 scores
    (unbounded, corpus-dependent) living on completely different scales,
    which would make directly averaging/comparing them meaningless.

    Parameters:
        rankings: a list of rankings, where each ranking is a list of
            chunk indices ordered best-to-worst by that particular
            search method.
        k: a smoothing constant (RRF standard default is 60) that reduces
            the influence of exact rank position for lower-ranked items,
            preventing any single ranking from dominating the fused result.

    Returns:
        A list of (chunk_index, fused_score) tuples, sorted best-to-worst
        by combined score.
    """
    fused_scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            # Each ranking contributes 1 / (k + rank + 1) to a chunk's total
            # score, so a better (lower) rank contributes more — this is the
            # core RRF formula.
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(query: str, chunks: list[str], metadatas: list[dict], bm25_index: dict, top_k: int = 3):
    """Combine semantic (Chroma vector) and keyword (BM25) search via reciprocal rank fusion."""
    query_embedding = model.encode(query)

    # Embeds and stores the corpus in Chroma only the first time; a no-op after that.
    ensure_indexed(chunks, metadatas)

    # RRF needs a full ranking from each method to fuse fairly, so ask
    # Chroma to rank every chunk rather than just a small top_k slice.
    vector_ranking = vector_search(query_embedding, top_k=len(chunks))

    bm25_ranking = [i for i, _ in bm25_search(query, bm25_index)]

    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

    return [(chunks[doc_id], score, metadatas[doc_id]["source"]) for doc_id, score in fused[:top_k]]
