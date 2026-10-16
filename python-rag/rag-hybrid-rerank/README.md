# rag-hybrid-rerank

Builds on `rag-from-scratch` by adding **hybrid retrieval** (vector + keyword search) and a **reranking** stage, plus a real evaluation harness to measure whether these additions actually help.

## Pipeline

1. **`chunker.py`** — same fixed-size chunking as the base stage.
2. **`embedder.py`** — same `SentenceTransformer` (`all-MiniLM-L6-v2`) bi-encoder embeddings.
3. **`bm25.py`** — classic keyword/lexical search: computes term frequency, inverse document frequency (IDF), and length normalization (BM25's `k1`/`b` constants) to score chunks purely on exact term overlap. Complements vector search, which can miss exact names/numbers/acronyms.
4. **`retriever.py`** — `hybrid_retrieve()` runs both vector search and BM25, then merges their two rankings using **Reciprocal Rank Fusion (RRF)**: instead of averaging incompatible score scales, it combines results by rank *position*, rewarding chunks both methods agree on.
5. **`reranker.py`** — a **cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the top ~15 hybrid-retrieved candidates by reading the query and each chunk *together* (more accurate than the bi-encoder, but too slow to run over the whole corpus), narrowing down to the final top 3.
6. **`generator.py`** — same grounded-prompt + local Ollama generation as the base stage, now fed the reranked top 3 chunks.
7. **`evaluation.py`** — a hand-labeled test set of (question, expected-answer-substring) pairs run through the full pipeline, scored by:
   - **Hit Rate@3** — % of questions where the right chunk landed in the top 3.
   - **MRR (Mean Reciprocal Rank)** — rewards the right chunk ranking *near the top*, not just anywhere in top-3.

## Results (12-question eval set against `sample.txt`)

**Hit Rate@3: 100%, MRR: 0.722**

## Running it

```bash
ollama serve   # if not already running
python generator.py      # interactive Q&A
python evaluation.py     # run the retrieval evaluation suite
```

## What this stage covers, on top of the base

- BM25 keyword search
- Reciprocal Rank Fusion (hybrid retrieval)
- Cross-encoder reranking (bi-encoder vs. cross-encoder tradeoff)
- Retrieval evaluation methodology (Hit Rate@k, MRR, ground-truth test sets)
