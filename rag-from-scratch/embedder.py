from pathlib import Path
from sentence_transformers import SentenceTransformer

from chunker import load_text, chunk_text

SCRIPT_DIR = Path(__file__).parent
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[str]):
    embeddings = model.encode(chunks)
    return embeddings
