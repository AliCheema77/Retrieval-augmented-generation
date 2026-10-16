# rag-query-transform

A copy of `rag-conversational`, evolved to add **query transformation** — generating alternative phrasings of a question before retrieval, instead of retrieving with only the one phrasing `rewrite_query()` produced.

## What changed from `rag-conversational`

### 1. Multi-query expansion (`query_expander.py`)
- `expand_query(query, num_variants=3)` makes one LLM call asking for `num_variants` alternative phrasings of the (already conversationally-rewritten) question, returning the original query plus the variants.
- The original query is always included alongside the variants — the fused ranking should never lose the literal phrasing the user actually meant, in case the LLM's rewrites drift.

### 2. Fusing across multiple phrasings (`retriever.py`)
- New `multi_query_retrieve()` sits alongside the existing `hybrid_retrieve()` (kept as the baseline for comparison, not removed).
- For each phrasing from `expand_query()`, it runs both vector search and BM25 search, producing 2 rankings per phrasing — with the default 3 variants + the original, that's **8 rankings total**.
- All 8 are fused with the same `reciprocal_rank_fusion()` used for the 2-ranking hybrid case — RRF already generalizes to any number of rankings, so nothing there needed to change. A chunk that shows up near the top across several phrasings/methods accumulates score from each one, which is the mechanism by which phrasing diversity is supposed to help.

### 3. Head-to-head evaluation (`evaluation.py`)
- `evaluate()` now takes a `retrieve_fn` parameter instead of hardcoding `hybrid_retrieve`, so the same test harness runs against either retriever.
- `__main__` runs the full multi-turn test set through both `hybrid_retrieve` and `multi_query_retrieve` back to back, printing Hit Rate@3/MRR for each.

## Results (same 6-turn / 3-conversation eval set as `rag-conversational`)

| Retriever | Hit Rate@3 | MRR |
|---|---|---|
| `hybrid_retrieve` | 100% | 0.917 |
| `multi_query_retrieve` | 100% | 0.917 |

**Honest finding:** identical results, at roughly 4x the retrieval cost (4 phrasings × 2 search methods per query, vs. 1 × 2). This is most likely a **ceiling effect**, not proof that query expansion doesn't help: the eval questions already contain strong, distinctive keywords ("PORTpass", "TechnoGenics", "remote work policy") that both BM25 and vector search find easily on their own, leaving no room to improve on an already-correct rank 1. Query expansion is expected to earn its keep on *harder* queries — vague phrasing, synonyms the corpus doesn't literally contain, or questions where the obvious keywords are misleading — which this eval set doesn't currently test. Confirming that would need a deliberately harder test set, not just this one.

## Running it

```bash
ollama serve   # if not already running
python generator.py      # interactive multi-turn Q&A, now using multi_query_retrieve
python evaluation.py     # compares hybrid_retrieve vs multi_query_retrieve
```

## What this stage covers, on top of `rag-conversational`

- Query transformation (multi-query expansion) as a pre-retrieval step
- Generalizing Reciprocal Rank Fusion from 2 rankings to N
- Designing a retrieval method as a swappable function (`retrieve_fn`) so evaluation can compare approaches head-to-head
- Recognizing a ceiling effect in an eval set — a technique can be sound but untested by questions that were already easy
