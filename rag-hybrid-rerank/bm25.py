import re                        # re: built-in regular-expression module, used here to split text into word tokens
import math                      # math: built-in module providing math.log() for the IDF (inverse document frequency) formula
from collections import Counter  # Counter: built-in dict subclass that counts hashable items (from the collections module)


def tokenize(text: str) -> list[str]:
    """
    Lowercase a string and split it into a list of word tokens.

    This is the shared tokenization step used both when building the BM25
    index and when scoring a query, so index terms and query terms are
    compared on equal footing (same casing, same word-splitting rules).

    Parameters:
        text: the raw text to tokenize.

    Returns:
        A list of lowercase word tokens.
    """
    # text.lower(): built-in str method, converts all characters to lowercase
    # so "Python" and "python" are treated as the same token.
    #
    # re.findall(r"\w+", ...): built-in `re` function that returns every
    # non-overlapping match of the pattern as a list. \w+ matches one or
    # more "word characters" (letters, digits, underscore), which
    # effectively splits the text into words and strips out punctuation/
    # whitespace.
    return re.findall(r"\w+", text.lower())


def build_bm25_index(chunks: list[str]) -> dict:
    """
    Precompute the corpus-wide statistics that BM25 scoring needs.

    BM25 scores a chunk against a query using: how long the chunk is
    relative to the average chunk, how often each query term appears in
    the chunk, and how rare/informative each term is across the whole
    corpus (IDF). All of those (except term frequency, computed later per
    query) can be calculated once up front — that's what this function does,
    so bm25_search() doesn't have to redo this work on every query.

    Parameters:
        chunks: list of raw text chunks making up the searchable corpus.

    Returns:
        A dict bundling everything bm25_score()/bm25_search() need:
            tokenized_chunks: each chunk pre-tokenized into words.
            doc_lengths: token count of each chunk.
            avg_doc_length: average token count across all chunks.
            idf: dict mapping each term to its inverse-document-frequency weight.
    """
    # Tokenize every chunk once up front (list comprehension — built-in
    # Python syntax for building a list by applying an expression to each
    # item of an iterable).
    tokenized_chunks = [tokenize(chunk) for chunk in chunks]

    # len(tokens): built-in function giving the number of tokens in each chunk,
    # i.e. that chunk's "document length" in BM25 terms.
    doc_lengths = [len(tokens) for tokens in tokenized_chunks]

    # sum(...) / len(...): built-in functions — average chunk length across
    # the whole corpus. Chunks longer than average get a small BM25 length
    # penalty; shorter ones get a small boost.
    avg_doc_length = sum(doc_lengths) / len(doc_lengths)

    # Counter(): built-in dict subclass used here to count, for each term,
    # in how many *distinct* chunks it appears at least once (document
    # frequency), not its total occurrence count.
    doc_freqs = Counter()
    for tokens in tokenized_chunks:
        # set(tokens): built-in function that de-duplicates the token list,
        # so a term appearing 5 times in one chunk only increments that
        # chunk's contribution to doc_freqs by 1, not 5.
        for term in set(tokens):
            doc_freqs[term] += 1

    num_docs = len(chunks)

    # IDF (inverse document frequency): terms that appear in very few chunks
    # get a high weight (they're distinctive/informative); terms that appear
    # in almost every chunk get a low (or negative) weight (they're common
    # and less useful for distinguishing relevance, e.g. "the", "and").
    # math.log(): built-in natural logarithm function, part of the standard
    # BM25 IDF formula.
    idf = {
        term: math.log((num_docs - freq + 0.5) / (freq + 0.5) + 1) for term, freq in doc_freqs.items()
    }

    # Bundle everything bm25_search() will need into one dict, so callers
    # only have to pass around this single "index" object.
    return {
        "tokenized_chunks": tokenized_chunks,
        "doc_lengths": doc_lengths,
        "avg_doc_length": avg_doc_length,
        "idf": idf
    }


def bm25_score(query_tokens, doc_tokens, doc_length, avg_doc_length, idf, k1=1.5, b=0.75) -> float:
    """
    Compute the BM25 relevance score of a single chunk against a query.

    Higher scores mean the chunk is more relevant. The formula rewards
    chunks that contain rare/important query terms (high IDF) frequently,
    while applying a diminishing-returns curve (via k1) so repeating a term
    100 times isn't 100x more relevant than once, and a length-normalization
    factor (via b) so long chunks aren't unfairly favored just for
    containing more words overall.

    Parameters:
        query_tokens: tokenized query terms.
        doc_tokens: tokenized terms of the chunk being scored.
        doc_length: number of tokens in this chunk.
        avg_doc_length: average chunk length across the corpus.
        idf: dict of term -> inverse document frequency weight.
        k1: controls how quickly term-frequency saturates (standard BM25
            tuning constant, 1.5 is a common default).
        b: controls how strongly chunk length is penalized/normalized
            (standard BM25 tuning constant, 0.75 is a common default).

    Returns:
        The BM25 score as a float (0.0 if no query terms are found).
    """
    # Counter(doc_tokens): built-in — counts how many times each word
    # appears in this specific chunk (term frequency).
    term_freqs = Counter(doc_tokens)
    score = 0.0
    for term in query_tokens:
        if term not in term_freqs:
            # Term never appears in this chunk at all — contributes nothing.
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
    """
    Rank every chunk in a prebuilt BM25 index against a query.

    Parameters:
        query: the raw query text.
        index: the dict returned by build_bm25_index().

    Returns:
        A list of (chunk_index, bm25_score) tuples for every chunk in the
        corpus, sorted from most to least relevant.
    """
    # Tokenize the query the same way chunks were tokenized when indexed.
    query_tokens = tokenize(query)
    scores = []
    # enumerate(): built-in function that pairs each item with its index,
    # so we know which chunk (by position) each score belongs to.
    for i, doc_tokens in enumerate(index["tokenized_chunks"]):
        score = bm25_score(
            query_tokens, doc_tokens,
            index["doc_lengths"][i], index["avg_doc_length"], index["idf"],
        )
        scores.append((i, score))
    # list.sort(): built-in method, sorted in place.
    #   key=lambda x: x[1] -> sort by the score (second tuple element)
    #   reverse=True        -> highest score first
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores
