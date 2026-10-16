from sentence_transformers import CrossEncoder

# Unlike the bi-encoder used for embeddings (which encodes the query and
# each chunk *separately* and compares them via cosine similarity), a
# cross-encoder reads the query and a chunk *together* in one forward
# pass, letting it directly judge how relevant that specific chunk is to
# that specific query — slower per pair, but typically more accurate.
# That's why it's used here as a second-pass reranker on a small
# shortlist rather than for searching the whole corpus.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[tuple[str, float]], top_k: int = 3) -> list[tuple[str, float]]:
    """
    Re-score candidate chunks with a cross-encoder, then keep only the
    true top-k most relevant ones. Meant to run after a cheaper retrieval
    step has already narrowed the corpus down to a small shortlist —
    reranking the whole corpus this way would be too slow.
    """
    pairs = [(query, chunk) for chunk, _score in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip([chunk for chunk, _score in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return reranked[:top_k]
