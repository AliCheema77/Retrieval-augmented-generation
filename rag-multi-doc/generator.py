from pathlib import Path        # Path: object-oriented filesystem paths (built-in pathlib)
import ollama                   # Python client for talking to the local Ollama server (ollama.generate, etc.)

from chunker import load_text, chunk_text
from retriever import hybrid_retrieve
from bm25 import build_bm25_index
from reranker import rerank

# __file__ is the path to this script; .parent gives its containing directory.
# Used so file paths (like sample.txt) work regardless of where the script is run from.
SCRIPT_DIR = Path(__file__).parent

# Name of the local model Ollama should use to generate answers.
# Must already be pulled locally via `ollama pull llama3.2:3b`.
OLLAMA_MODEL = "llama3.2:3b"

# Fixed instruction sent to the LLM on every call, telling it how to behave:
# restrict itself to the provided context and avoid hallucinating an answer.
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "context provided below. If the answer isn't in the context, say you "
    "don't know — do not make anything up."
)


def build_prompt(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    """
    Turn the retrieved chunks + user question into the final text prompt
    that gets sent to the LLM.

    Parameters:
        query: the raw question typed by the user.
        retrieved_chunks: a list of (chunk_text, relevance_score) tuples,
            already selected and ranked by the retriever/reranker pipeline.
            We only need the chunk text here — the score was just used
            earlier to decide which chunks made the cut.

    Returns:
        A single string combining all the context chunks and the question,
        formatted so the LLM can clearly tell context apart from the question.
    """
    # "\n\n---\n\n".join(...) : built-in str.join() method — concatenates all
    # chunk strings into one string, inserting "\n\n---\n\n" between each pair
    # as a visual separator so the model can distinguish separate chunks.
    #
    # "for chunk, _score in retrieved_chunks" unpacks each (chunk, score) tuple;
    # we only use `chunk` and throw away the score (conventionally named
    # `_score` to signal "intentionally unused").
    context = "\n\n---\n\n".join(chunk for chunk, _score in retrieved_chunks)

    # f-string (built-in Python formatted string literal) to interleave the
    # context and the question into the final prompt text.
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    """
    Send the assembled prompt to the local Ollama model and return its answer.

    Parameters:
        query: the raw question typed by the user.
        retrieved_chunks: the top-ranked (chunk, score) pairs to use as context.

    Returns:
        The plain-text answer string produced by the LLM.
    """
    # Build the full prompt text (context + question) using the helper above.
    prompt = build_prompt(query, retrieved_chunks)

    # ollama.generate(): built-in function from the `ollama` Python package.
    # Sends a one-off (non-chat, non-streaming) generation request to the
    # local Ollama server (http://localhost:11434) running in the background.
    #   model  -> which local model to use
    #   system -> the system-level instruction/behavior for the model
    #   prompt -> the user-facing content (context + question) to respond to
    # Returns a dict with metadata plus the generated text under "response".
    response = ollama.generate(
        model=OLLAMA_MODEL,
        system=SYSTEM_PROMPT,
        prompt=prompt,
    )

    # Extract just the generated answer text from the response dict.
    return response["response"]


# This block only runs when the script is executed directly
# (e.g. `python generator.py`), not when imported as a module elsewhere.
if __name__ == "__main__":
    # Build the full path to the sample document living next to this script.
    sample_path = SCRIPT_DIR / "documents"

    # load_text(): reads the raw text content of the sample file (from chunker.py).
    text = load_text(sample_path)

    # chunk_text(): splits the raw text into smaller overlapping/non-overlapping
    # pieces ("chunks") suitable for embedding and retrieval (from chunker.py).
    chunks = chunk_text(text)

    # input(): built-in function that pauses execution, prints the given
    # prompt string, and waits for the user to type a line and press Enter;
    # returns whatever the user typed as a string.
    query = input(f"Ask your questions here\n")

    # build_bm25_index(): builds a BM25 keyword-search index over the chunks,
    # used for classic lexical (keyword) matching alongside vector search
    # (from bm25.py).
    bm25_index = build_bm25_index(chunks)

    # hybrid_retrieve(): combines vector similarity search and BM25 keyword
    # search to pull the top_k=15 most relevant candidate chunks for the
    # query (from retriever.py). Returns (chunk, score) tuples.
    candidate_chunks = hybrid_retrieve(query, chunks, bm25_index, top_k=15)

    # rerank(): takes the 15 candidates and re-scores/re-orders them using a
    # (presumably more accurate but slower) reranking model, keeping only the
    # top_k=3 best matches to actually feed into the LLM (from reranker.py).
    retrieved_chunks = rerank(query, candidate_chunks, top_k=3)

    # Generate the final answer using only these top 3 chunks as context.
    answer = generate_answer(query, retrieved_chunks)

    # print(): built-in function that writes text to standard output (the terminal).
    print(f"Query: {query}\n")
    print(f"Answer:\n{answer}")
