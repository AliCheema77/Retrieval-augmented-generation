import ollama

OLLAMA_MODEL = "llama3.2:3b"

REWRITE_SYSTEM_PROMPT = (
    "Given a chat history and a follow-up question, rewrite the follow-up "
    "question into a standalone question that can be understood without the "
    "chat history. Do not answer the question — only rewrite it. If the "
    "question is already standalone, return it unchanged."
)


def rewrite_query(query: str, history: list[tuple[str, str]]) -> str:
    """Resolve references in `query` (e.g. "that", "it") using prior turns."""
    if not history:
        return query
    
    history_text = "\n".join(f"User: {q}\nAssistant: {a}" for q, a in history)
    prompt = f"Chat History:\n{history_text}\n\nFollow-up question: {query}\n\nStandalone question:"
    
    response = ollama.generate(
        model=OLLAMA_MODEL,
        system=REWRITE_SYSTEM_PROMPT,
        prompt=prompt,
    )
    return response["response"].strip()