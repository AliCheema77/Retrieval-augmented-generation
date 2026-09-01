from pathlib import Path        # Path: object-oriented filesystem paths (built-in pathlib)
import ollama                   # Python client for talking to the local Ollama server (ollama.generate, etc.)

from chunker import load_documents, chunk_documents
from retriever import hybrid_retrieve
from bm25 import build_bm25_index
from reranker import rerank
from query_rewriter import rewrite_query

# __file__ is the path to this script; .parent gives its containing directory.
# Used so file paths (like documents/) work regardless of where the script is run from.
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


def build_prompt(query: str, retrieved_chunks: list[tuple[str, float, str]]) -> str:
    """
    Turn the retrieved chunks + user question into the final text prompt
    that gets sent to the LLM.

    Parameters:
        query: the raw question typed by the user.
        retrieved_chunks: a list of (chunk_text, relevance_score, source)
            tuples, already selected and ranked by the retriever/reranker
            pipeline. The score isn't needed here; the source (which
            document the chunk came from) is included so the model — and
            you — can see where each piece of context came from.

    Returns:
        A single string combining all the context chunks and the question,
        formatted so the LLM can clearly tell context apart from the question.
    """
    context = "\n\n---\n\n".join(f"[Source: {source}]\n{chunk}" for chunk, _score, source in retrieved_chunks)
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate_answer(query: str, retrieved_chunks: list[tuple[str, float, str]]) -> str:
    """
    Send the assembled prompt to the local Ollama model and return its answer.

    Parameters:
        query: the raw question typed by the user.
        retrieved_chunks: the top-ranked (chunk, score, source) triples to use as context.

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
    documents = load_documents(SCRIPT_DIR / "documents")
    chunks, metadatas = chunk_documents(documents)
    bm25_index = build_bm25_index(chunks)

    # Grows one (raw_query, answer) tuple per turn. rewrite_query() reads
    # this to resolve references in later questions (e.g. "it", "there").
    history = []

    while True:
        query = input("Ask your questions here (or type 'exit' to quit)\n")
        if query.lower() == "exit":
            break

        standalone_query = rewrite_query(query, history)
        if standalone_query != query:
            print(f"(rewritten as: {standalone_query})")

        # Retrieval, reranking, and generation all run on the *rewritten*
        # query, not the raw one — that's the whole point of rewriting it.
        candidate_chunks = hybrid_retrieve(standalone_query, chunks, metadatas, bm25_index, top_k=15)
        retrieved_chunks = rerank(standalone_query, candidate_chunks, top_k=3)
        answer = generate_answer(standalone_query, retrieved_chunks)

        print(f"\nAnswer:\n{answer}\n")

        # Store the raw query (not the rewritten one) so the history stays
        # a readable log of what was actually typed.
        history.append((query, answer))
