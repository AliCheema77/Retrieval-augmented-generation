from sentence_transformers import CrossEncoder  # CrossEncoder: loads a pretrained query-document relevance scoring model (sentence-transformers library)

# Load the cross-encoder model once at import time (not inside a function),
# so the relatively slow model loading only happens a single time no matter
# how many times rerank() is called afterward.
#
# Unlike the bi-encoder used for embeddings (which encodes the query and
# each chunk *separately* into vectors and compares them via cosine
# similarity), a cross-encoder reads the query and a chunk *together* in
# one forward pass, letting it directly judge how relevant that specific
# chunk is to that specific query — slower per pair, but typically more
# accurate, which is why it's used here as a second-pass reranking step
# on a small shortlist rather than for searching the whole corpus.
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, candidates: list[tuple[str, float, str]], top_k: int = 3) -> list[tuple[str, float, str]]:
    """
    Re-score a shortlist of candidate chunks using a cross-encoder, then
    keep only the true top-k most relevant ones.

    This is meant to run *after* an initial, cheaper retrieval step (e.g.
    hybrid_retrieve()) has already narrowed the whole corpus down to a
    small set of candidates — reranking every chunk in a large corpus with
    a cross-encoder would be too slow.

    Parameters:
        candidates: list of (chunk_text, prior_score, source) tuples from
            the earlier retrieval stage. The prior_score is discarded here
            since the cross-encoder produces its own, more reliable score;
            source (which document the chunk came from) is carried through
            unchanged so it can be shown alongside the final answer.
        top_k: how many chunks to keep after reranking.

    Returns:
        A list of (chunk_text, cross_encoder_score, source) tuples, sorted
        best match first, truncated to top_k.
    """
    pairs = [(query, chunk) for chunk, _score, _source in candidates]
    scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(
            [chunk for chunk, _score, _source in candidates],
            scores,
            [source for _chunk, _score, source in candidates],
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    return reranked[:top_k]
