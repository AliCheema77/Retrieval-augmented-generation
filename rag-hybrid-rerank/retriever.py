import numpy as np              # numpy: numerical library used here for vector dot product and vector norm (magnitude)
from pathlib import Path        # Path: object-oriented filesystem paths (built-in pathlib)

from chunker import load_text, chunk_text
from embedder import model, embed_chunks
from bm25 import build_bm25_index, bm25_search

# __file__ is the path to this script; .parent gives its containing directory.
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
    # np.dot(): NumPy built-in function computing the dot product of two
    # vectors (sum of elementwise products) — measures how much the vectors
    # "align."
    dot_product = np.dot(a, b)
    # np.linalg.norm(): NumPy built-in function computing the Euclidean
    # length (magnitude) of a vector.
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    # Dividing the dot product by both magnitudes normalizes out vector
    # length, leaving only the directional (angular) similarity.
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
    # model.encode(): SentenceTransformer method (from embedder.py's shared
    # model) — embeds the query into the same vector space as the chunks,
    # so they can be compared directly.
    query_embedding = model.encode(query)

    scores = []
    # enumerate(): built-in function pairing each embedding with its index,
    # so we can map a score back to the original chunk text afterward.
    for i, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, i))

    # list.sort(): built-in method, sorted in place.
    #   reverse=True         -> highest similarity first
    #   key=lambda x: x[0]   -> sort by the score (first tuple element)
    scores.sort(reverse=True, key=lambda x: x[0])

    top_chunks = []
    # scores[:top_k]: built-in list slicing — keep only the first top_k
    # (highest-scoring) entries.
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
        # enumerate(): built-in function — `rank` is the position (0 = best)
        # of `doc_id` within this particular ranking.
        for rank, doc_id in enumerate(ranking):
            # dict.get(key, default): built-in dict method — returns the
            # current fused score for doc_id, or 0.0 if it hasn't been seen
            # in any ranking yet. Each ranking contributes 1/(k + rank + 1)
            # to a document's total score: a better (lower) rank contributes
            # more.
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    # sorted(): built-in function that returns a new sorted list without
    # mutating the input.
    #   .items()            -> (doc_id, score) pairs from the dict
    #   key=lambda x: x[1]  -> sort by the fused score
    #   reverse=True        -> highest score first
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(query: str, chunks: list[str], chunk_embeddings, bm25_index: dict, top_k: int = 3):
    """
    Retrieve the most relevant chunks using both semantic (vector) search
    and keyword (BM25) search, then merge the two rankings.

    Combining both catches cases either method alone would miss: vector
    search finds chunks that are conceptually related even without shared
    wording, while BM25 reliably finds chunks containing exact terms
    (names, acronyms, numbers) that embeddings can sometimes blur together.

    Parameters:
        query: the raw question text.
        chunks: the original text chunks.
        chunk_embeddings: precomputed embedding vector for each chunk.
        bm25_index: the prebuilt BM25 index (from build_bm25_index()).
        top_k: how many final merged chunks to return.

    Returns:
        A list of (chunk_text, fused_score) tuples, best match first.
    """
    # Embed the query once, reused for the vector-ranking step below.
    query_embedding = model.encode(query)

    # List comprehension: compute cosine similarity between the query and
    # every chunk embedding, keeping track of each chunk's original index.
    vector_scores = [(i, cosine_similarity(query_embedding, e)) for i, e in enumerate(chunk_embeddings)]
    # Sort chunks by similarity score, best first.
    vector_scores.sort(key=lambda x: x[1], reverse=True)
    # Extract just the ranked list of chunk indices (RRF only needs rank
    # order, not the raw similarity scores).
    vector_ranking = [i for i, _ in vector_scores]

    # bm25_search() already returns (index, score) tuples sorted best-first;
    # pull out just the ranked indices the same way as above.
    bm25_ranking = [i for i, _ in bm25_search(query, bm25_index)]

    # Merge both rankings into one combined ranking via reciprocal rank fusion.
    fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])

    # fused[:top_k]: built-in list slicing — keep only the top_k best fused
    # results, then map each chunk index back to its actual chunk text.
    return [(chunks[doc_id], score) for doc_id, score in fused[:top_k]]
