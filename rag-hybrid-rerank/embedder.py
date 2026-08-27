from pathlib import Path                          # Path: object-oriented filesystem paths (built-in pathlib)
from sentence_transformers import SentenceTransformer  # SentenceTransformer: loads a pretrained embedding model (sentence-transformers library)

from chunker import load_text, chunk_text

# __file__ is the path to this script; .parent gives its containing directory.
SCRIPT_DIR = Path(__file__).parent

# Load the "all-MiniLM-L6-v2" pretrained sentence-embedding model once at
# import time (not inside a function), so the (relatively slow) model
# loading only happens a single time no matter how many times embed_chunks()
# is called afterward. This model turns text into 384-dimensional vectors.
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks: list[str]):
    """
    Convert a list of text chunks into numeric vector embeddings.

    Embeddings place semantically similar pieces of text close together in
    vector space, which is what lets us later find "chunks similar in
    meaning to the query" via vector similarity search, rather than only
    exact keyword matching.

    Parameters:
        chunks: list of text chunks to embed.

    Returns:
        A NumPy array of shape (num_chunks, embedding_dim), one embedding
        vector per input chunk.
    """
    # model.encode(): built-in method from SentenceTransformer that runs the
    # model's forward pass over a batch of text and returns their embedding
    # vectors as a NumPy array.
    embeddings = model.encode(chunks)
    return embeddings


# This block only runs when the script is executed directly
# (e.g. `python embedder.py`), not when imported as a module elsewhere.
if __name__ == "__main__":
    # Build the full path to the sample document living next to this script.
    sample_path = SCRIPT_DIR / "sample.txt"

    # Read the raw file contents into a single string.
    text = load_text(sample_path)

    # Split that string into overlapping chunks using the default sizes.
    chunks = chunk_text(text)

    # Convert every chunk into its embedding vector.
    embeddings = embed_chunks(chunks)

    # print(): built-in function that writes text to standard output (the terminal).
    print(f"Number of chunks: {len(chunks)}")
    # .shape: NumPy array attribute giving (rows, columns) — here
    # (number_of_chunks, embedding_dimension).
    print(f"Embedding shape: {embeddings.shape}")
    # embeddings[0][:10]: NumPy slicing — take the first embedding vector,
    # then just its first 10 numbers, so the printout stays short.
    print(f"First embedding (first 10 values):\n{embeddings[0][:10]}")
