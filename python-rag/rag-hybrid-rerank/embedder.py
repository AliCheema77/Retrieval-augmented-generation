from pathlib import Path
from sentence_transformers import SentenceTransformer

from chunker import load_text, chunk_text

SCRIPT_DIR = Path(__file__).parent

# Loaded once at import time (not inside a function) so the relatively slow
# model loading only happens once no matter how many times embed_chunks()
# is called afterward.
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[str]):
    """Convert each text chunk into a numeric vector so semantically similar chunks end up close together in vector space."""
    embeddings = model.encode(chunks)
    return embeddings


if __name__ == "__main__":
    sample_path = SCRIPT_DIR / "sample.txt"
    text = load_text(sample_path)
    chunks = chunk_text(text)

    embeddings = embed_chunks(chunks)

    print(f"Number of chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"First embedding (first 10 values):\n{embeddings[0][:10]}")
