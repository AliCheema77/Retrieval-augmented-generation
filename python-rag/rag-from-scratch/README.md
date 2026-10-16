# rag-from-scratch

The simplest possible RAG (Retrieval-Augmented Generation) pipeline, built in plain Python with no frameworks (no LangChain/LlamaIndex), to learn the core mechanics end to end.

## Pipeline

1. **`chunker.py`** — reads a text file and splits it into fixed-size, overlapping character chunks (`chunk_size` / `overlap`), so retrieval works on focused passages instead of the whole document.
2. **`embedder.py`** — converts each chunk into a numeric vector using a local `SentenceTransformer` model (`all-MiniLM-L6-v2`), so semantically similar text ends up close together in vector space.
3. **`retriever.py`** — embeds the user's query and ranks chunks against it by **cosine similarity**, returning the top-k most relevant chunks.
4. **`generator.py`** — assembles the retrieved chunks + question into a prompt, with a system prompt instructing the model to answer only from the given context, and sends it to a local LLM via **Ollama**.

## Running it

Requires Ollama running locally with the model pulled:
```bash
ollama serve
ollama pull llama3.2:3b
```

Then, from this directory:
```bash
python generator.py
```
It will prompt for a question, retrieve relevant chunks from `sample.txt`, and print the LLM's answer.

## What this stage covers

- Fixed-size chunking
- Bi-encoder embeddings
- Cosine similarity vector search
- Grounded prompt construction (anti-hallucination system prompt)
- Local LLM inference via Ollama

No evaluation, no keyword search, no reranking, no persistence — those are added in later stages (`rag-hybrid-rerank`, `rag-vector-db`).
