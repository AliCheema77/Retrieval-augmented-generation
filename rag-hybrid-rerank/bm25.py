import re
import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens for keyword-based (BM25) matching."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: list[str]) -> dict:
    """
    Precompute the corpus-wide statistics that BM25 scoring needs: doc
    lengths, average doc length, and each term's inverse document
    frequency (IDF). Doing this once up front means bm25_search() doesn't
    have to redo this work on every query.
    """
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    doc_lengths = [len(tokens) for tokens in tokenized_chunks]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)

    # Document frequency: how many distinct chunks each term appears in at
    # least once (not its total occurrence count) — that's why we iterate
    # over set(tokens) rather than tokens itself.
    doc_freqs = Counter()
    for tokens in tokenized_chunks:
        for term in set(tokens):
            doc_freqs[term] += 1

    num_docs = len(chunks)

    # IDF: terms that appear in very few chunks get a high weight (they're
    # distinctive/informative); terms that appear in almost every chunk get
    # a low weight (common words like "the", "and" aren't useful for
    # distinguishing relevance).
    idf = {
        term: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in doc_freqs.items()
    }

    return {
        "tokenized_chunks": tokenized_chunks,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "idf": idf
    }


def bm25_score(query_tokens, doc_tokens, doc_length, avg_doc_length, idf, k1=1.5, b=0.75) -> float:
    """
    Score how relevant one chunk is to the query terms.

    Rewards chunks containing rare/important query terms (high IDF)
    frequently, while applying a diminishing-returns curve (k1) so
    repeating a term 100 times isn't 100x more relevant than once, and a
    length-normalization factor (b) so long chunks aren't unfairly favored
    just for containing more words overall.
    """
    term_freqs = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in term_freqs:
            continue
        f = term_freqs[term]
        # Standard BM25 term-score formula:
        #   numerator   = idf(term) * f * (k1 + 1)
        #   denominator = f + k1 * (1 - b + b * doc_length / avg_doc_length)
        numerator = idf.get(term, 0.0) * f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * doc_length / avg_doc_length)
        score += numerator / denominator
    return score


def bm25_search(query: str, index: dict) -> list[tuple[int, float]]:
    """Rank every chunk in the index against the query by BM25 score, highest first."""
    query_tokens = tokenize(query)
    scores = []
    for i, doc_tokens in enumerate(index["tokenized_chunks"]):
        score = bm25_score(
            query_tokens, doc_tokens,
            index["doc_lengths"][i], index["avg_doc_length"], index["idf"],
        )
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
