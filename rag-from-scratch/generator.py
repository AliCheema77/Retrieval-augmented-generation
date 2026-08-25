from pathlib import Path
import ollama

from chunker import load_text, chunk_text
from embedder import embed_chunks
from retriever import retrieve

SCRIPT_DIR = Path(__file__).parent
OLLAMA_MODEL = "llama3.2:3b"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the answer isn't in the context, say you "
    "don't know — do not make anything up."
)


def build_prompt(query:str, retrieved_chunks: list[tuple[str, float]]) -> str:
    context = "\n\n---\n\n".join(chunk for chunk, _score in retrieved_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"

def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    prompt = build_prompt(query, retrieved_chunks)

    response = ollama.generate(
        model=OLLAMA_MODEL,
        system=SYSTEM_PROMPT,
        prompt=prompt,
    )

    return response["response"]


if __name__ == "__main__":
    sample_path = SCRIPT_DIR / "sample.txt"
    text = load_text(sample_path)
    chunks = chunk_text(text)
    chunk_embeddings = embed_chunks(chunks)

    query = "Did candidate work with docker?"
    retrieved_chunks = retrieve(query, chunks, chunk_embeddings)

    answer = generate_answer(query, retrieved_chunks)

    print(f"Query: {query}\n")
    print(f"Answer:\n{answer}")