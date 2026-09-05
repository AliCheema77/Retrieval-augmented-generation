import ollama

OLLAMA_MODEL = "llama3.2:3b"

GRADE_SYSTEM_PROMPT = (
    "You are grading whether a piece of retrieved text is relevant to a "
    "question. Answer with exactly one word: 'yes' if the text contains "
    "information that helps answer the question, or 'no' if it doesn't"
)


def grade_relevance(query: str, chunk: str) -> bool:
    """Ask the LLM whether `chunk` actually helps answer `query`."""
    prompt = f"Question: {query}\n\nRetrieved text:\n{chunk}\n\nIs this text relevant? Answer yes or no."
    
    response = ollama.generate(
        model=OLLAMA_MODEL,
        system=GRADE_SYSTEM_PROMPT,
        prompt=prompt
    )
    
    return response["response"].stript().lower().startswith("yes")