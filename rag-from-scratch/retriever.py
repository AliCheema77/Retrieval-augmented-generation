import numpy as np
from pathlib import Path

from chunker import load_text, chunk_text
from embedder import model, embed_chunks


SCRIPT_DIR = Path(__file__).parent


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def retrieve(query: str, chunks: list[str], chunk_embeddings, top_k: int = 3):
    query_embedding = model.encode(query)

    scores = []
    for i, chunk_embedding in enumerate(chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        scores.append((score, i))

    scores.sort(reverse=True, key=lambda x: x[0])

    top_chunks = []
    for score, i in scores[:top_k]:
        top_chunks.append((chunks[i], score))

    return top_chunks


if __name__ == "__main__":
    sample_path = SCRIPT_DIR / "sample.txt"
    text = load_text(sample_path)
    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)

    query = "What are backend skills?"
    results = retrieve(query, chunks, chunk_embeddings)

    for chunk, score in results:
        print(f"Score: {score:.4f}")
        print(f"Chunk: {chunk}\n")