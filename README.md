# Retrieval-augmented-generation

A step-by-step, from-scratch exploration of RAG (Retrieval-Augmented Generation), built in plain Python with no frameworks, progressing from a naive pipeline to a hybrid-search-and-rerank pipeline with persistent vector storage.

Each subfolder is a self-contained stage — see its own README for details.

| Stage | Folder | Adds |
|---|---|---|
| 1. Naive RAG | [`rag-from-scratch/`](rag-from-scratch/README.md) | Chunking, embeddings, cosine similarity retrieval, local LLM generation via Ollama |
| 2. Hybrid + rerank | [`rag-hybrid-rerank/`](rag-hybrid-rerank/README.md) | BM25 keyword search, Reciprocal Rank Fusion, cross-encoder reranking, retrieval evaluation (Hit Rate@k, MRR) |
| 3. Vector DB | [`rag-vector-db/`](rag-vector-db/README.md) | Persistent storage via Chroma (HNSW ANN search), chunking-strategy experimentation |

All stages use a local Ollama model (`llama3.2:3b`) for generation — no external API calls or keys required.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

ollama serve
ollama pull llama3.2:3b
```

Then `cd` into whichever stage you want to run and see its README for usage.
