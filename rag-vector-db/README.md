# rag-vector-db

A copy of `rag-hybrid-rerank`, evolved to add **persistent vector storage** (via Chroma) and an experiment in **chunking strategy**, so retrieval no longer depends on recomputing everything from memory every run.

## What changed from `rag-hybrid-rerank`

### 1. Persistent vector storage (`vectorstore.py`)
- Uses Chroma's `PersistentClient`, writing its index to `chroma_store/` on disk instead of keeping embeddings only in memory.
- Chroma searches using its own approximate nearest-neighbor index (**HNSW**) instead of the manual brute-force cosine-similarity loop used in earlier stages — this is what lets it scale to far more chunks.
- `ensure_indexed(chunks)` embeds and stores the corpus **only the first time** (checked via `collection.count() > 0`) — on every later run, embedding is skipped entirely, not just re-storage. This removed the `chunk_embeddings` parameter from `hybrid_retrieve()` altogether, since Chroma is now the single source of truth for the corpus's vectors.

**Gotcha to remember:** if you change `sample.txt` or the chunking logic, you must delete `chroma_store/` before the next run — otherwise `ensure_indexed()` will see a non-empty collection and keep serving the *old*, stale chunks.
```bash
rm -rf chroma_store
```

### 2. Chunking strategy experiment (`chunker.py`)
- Replaced fixed-size character chunking with **line-boundary-aware chunking**: consecutive whole lines are packed together up to `chunk_size`, never splitting a line mid-way (falling back to a hard character cut only if a single line alone exceeds `chunk_size`).
- Swept `chunk_size`/`overlap` combinations (200 through 1000) against the evaluation suite; settled on **`chunk_size=700, overlap=70`** — the best-performing line-based config (Hit Rate@3: 100%, MRR: 0.708).
- **Honest finding:** this doesn't clearly beat the original character-based chunking from `rag-hybrid-rerank` (MRR 0.722) — with only 12 eval questions, a ~0.02–0.04 MRR gap is within sampling noise, not a proven improvement. Line-based chunking is architecturally sound (it avoids ever slicing through a sentence), but on this specific small document, it's statistically a wash rather than a decisive win.

## Running it

```bash
ollama serve   # if not already running
python generator.py      # interactive Q&A
python evaluation.py     # run the retrieval evaluation suite
```

## What this stage covers, on top of `rag-hybrid-rerank`

- Vector database persistence (Chroma)
- Approximate nearest-neighbor search (HNSW) vs. brute-force cosine loops
- Avoiding redundant computation across runs
- Hyperparameter sweeping methodology, and recognizing when a result is genuinely inconclusive rather than forcing a "winner"
