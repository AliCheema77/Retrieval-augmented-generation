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


def rerank(query: str, candidates: list[tuple[str, float]], top_k: int = 3) -> list[tuple[str, float]]:
    """
    Re-score a shortlist of candidate chunks using a cross-encoder, then
    keep only the true top-k most relevant ones.

    This is meant to run *after* an initial, cheaper retrieval step (e.g.
    hybrid_retrieve()) has already narrowed the whole corpus down to a
    small set of candidates — reranking every chunk in a large corpus with
    a cross-encoder would be too slow.

    Parameters:
        candidates: list of (chunk_text, prior_score) tuples from the
            earlier retrieval stage. The prior_score is discarded here since
            the cross-encoder produces its own, more reliable relevance score.
        top_k: how many chunks to keep after reranking.

    Returns:
        A list of (chunk_text, cross_encoder_score) tuples, sorted best
        match first, truncated to top_k.
    """
    # Build a (query, chunk) pair for every candidate — this is the input
    # format CrossEncoder.predict() expects, since it scores each pair
    # jointly rather than encoding query and chunk independently.
    # "for chunk, _score in candidates" unpacks each tuple, discarding the
    # earlier stage's score (named `_score` to mark it as intentionally unused).
    pairs = [(query, chunk) for chunk, _score in candidates]

    # cross_encoder.predict(): CrossEncoder built-in method — runs the model
    # over every (query, chunk) pair and returns a relevance score for each.
    scores = cross_encoder.predict(pairs)

    # zip(): built-in function that pairs up two equal-length sequences
    # element-by-element — here, each chunk's text with its new
    # cross-encoder score.
    #
    # sorted(): built-in function returning a new sorted list.
    #   key=lambda x: x[1]  -> sort by the cross-encoder score
    #   reverse=True        -> highest (most relevant) score first
    reranked = sorted(
        zip([chunk for chunk, _score in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    # reranked[:top_k]: built-in list slicing — keep only the top_k best results.
    return reranked[:top_k]
