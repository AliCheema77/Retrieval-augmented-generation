from pathlib import Path
import chromadb

from embedder import embed_chunks

SCRIPT_DIR = Path(__file__).parent
CHROMA_DIR = SCRIPT_DIR / "chroma_store"

# Chroma's PersistentClient writes its index to disk at CHROMA_DIR instead of
# keeping it only in memory, so the embeddings survive between script runs
# and don't need to be recomputed and re-uploaded every time.
client = chromadb.PersistentClient(path=str(CHROMA_DIR))

# A "collection" is Chroma's equivalent of a table: it stores each chunk's
# text, embedding, and id together, and knows how to search itself.
# get_or_create_collection loads it if it already exists on disk from a
# previous run, or creates an empty one on the first run.
collection = client.get_or_create_collection(name="chunks")


def ensure_indexed(chunks: list[str]) -> None:
    """Embed and store chunks in Chroma, but only the first time — skip entirely if already indexed."""
    if collection.count() > 0:
        return
    chunk_embeddings = embed_chunks(chunks)
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=chunk_embeddings.tolist(),
    )


def vector_search(query_embedding, top_k: int) -> list[int]:
    """Return the top_k chunk indices closest to the query embedding, ranked best-first."""
    # Chroma searches using its own approximate nearest-neighbor index
    # (HNSW) instead of comparing against every embedding one by one, which
    # is what lets it scale to far more chunks than a manual loop would.
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    # results["ids"] holds one list of matches per query we sent; since we
    # only sent one, we read index [0]. Ids were stored as strings, so they
    # need converting back to int to match the chunks list's indices.
    return [int(doc_id) for doc_id in results["ids"][0]]
