# rag-conversational

A copy of `rag-multi-doc`, evolved to handle **multi-turn conversations** — follow-up questions that depend on earlier chat history — instead of a single one-shot question per run.

## What changed from `rag-multi-doc`

### 1. Query rewriting (`query_rewriter.py`)
- `rewrite_query(query, history)` resolves references in a follow-up question (e.g. "that", "it", "there") into a standalone question, using an LLM call over the accumulated `(question, answer)` chat history.
- If there's no history yet (first turn), it's a no-op — the query is returned unchanged, no LLM call made.
- The rewritten (standalone) query — not the raw one — is what actually flows into `hybrid_retrieve()`, `rerank()`, and `generate_answer()`. The raw query is kept only for what's stored in `history` and shown to the user.

### 2. An actual conversation loop (`generator.py`)
- The old `__main__` block asked one question and exited. It's now a `while True` loop that keeps asking, accumulating `history` after every turn, until you type `exit`.

### 3. Multi-turn evaluation (`evaluation.py`)
- `TEST_CASES` changed from a flat list of independent questions to a list of short **conversations** — each a sequence of turns where a later turn refers back to an earlier one (e.g. "How much did **it** improve data retrieval efficiency by?" after "What is PORTpass?").
- `evaluate()` now runs each conversation with its own `history`, calling `rewrite_query()` before retrieval on every turn — and calls `generate_answer()` to produce the real answer that feeds into the *next* turn's rewrite, matching what actually happens in the live loop rather than faking history with the expected substring.

## Results (6-turn / 3-conversation eval set)

**Hit Rate@3: 100%, MRR: 0.917**

Every follow-up turn resolved its reference correctly and landed at rank 1 — e.g. "How much did **it** improve data retrieval efficiency by?" rewrote to "How much did **PORTpass** improve data retrieval efficiency?" First-turn questions pass through `rewrite_query()` unchanged, as expected with no prior history.

## Running it

```bash
ollama serve   # if not already running
python generator.py      # interactive multi-turn Q&A (type 'exit' to quit)
python evaluation.py     # run the multi-turn retrieval evaluation suite
```

## What this stage covers, on top of `rag-multi-doc`

- Query rewriting to resolve conversational references before retrieval
- Maintaining chat history across turns
- Designing an evaluation set for multi-turn behavior, not just single-shot retrieval
