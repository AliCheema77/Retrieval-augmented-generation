import re
import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into word tokens for keyword-based (BM25) matching."""
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: list[str]) -> dict:
    """Precompute per-corpus statistics (doc lengths, term document-frequencies, IDF) that BM25 scoring needs for every chunk."""
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]
    doc_lengths = [len(tokens) for tokens in tokenized_chunks]
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)
    
    doc_freqs = Counter()
    for tokens in tokenized_chunks:
        for term in set(tokens):
            doc_freqs[term] += 1
    
    num_docs = len(chunks)
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
    """Score how relevant one chunk is to the query terms, rewarding rare/important terms and penalizing overly long chunks."""
    term_freqs = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in term_freqs:
            continue
        f = term_freqs[term]
        numerator = idf.get(term, 0.0) * f * (k1 + 1)
        denominator = f + k1 * (1 -b + b * doc_length / avg_doc_length)
        score += numerator /denominator
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
