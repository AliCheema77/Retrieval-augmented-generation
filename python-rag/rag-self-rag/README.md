# rag-self-rag

A copy of `rag-query-transform`, to be evolved to add **Self-RAG / Corrective RAG** — grading retrieved chunks for actual relevance before generating an answer, and correcting course (reformulating and retrying) when nothing relevant comes back, instead of generating from whatever retrieval happened to return.

## Planned changes from `rag-query-transform`

- Add `relevance_grader.py`: `grade_relevance()` uses an LLM call to judge whether a single retrieved chunk is actually relevant to the question.
- Filter retrieved/reranked chunks through the grader before generation.
- If none pass grading, trigger a corrective step (reformulate the query and retry retrieval once) rather than generating an answer from irrelevant context.
- Everything else (chunking, persistence, source attribution, conversational rewriting, query expansion) carries over unchanged.

*(This README is a placeholder until the grading/correction logic is built.)*

## Running it

```bash
ollama serve   # if not already running
python generator.py      # interactive multi-turn Q&A
python evaluation.py     # run the retrieval evaluation suite
```
