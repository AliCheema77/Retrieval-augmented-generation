import ollama

OLLAMA_MODEL = "llama3.2:3b"


EXPAND_SYSTEM_PROMPT = (
    "Given a user's question, write 3 alternative phrasings of it that use "
    "different wording or emphasis, to help find relevant information from "
    "different angles. Return exactly 3 alternatives questions, one per line, with no "
    "numbering, bullets, or extra commentary."
)


def expand_query(query: str, num_variants: int=3) -> list[str]:
    """Generate alternative phrasings of `query` to widen retrieval coverage."""
    response = ollama.generate(
        model=OLLAMA_MODEL,
        system=EXPAND_SYSTEM_PROMPT,
        prompt=query
    )
    
    # The model is asked for one variant per line — split and drop blanks,
    # then cap at num_variants in case it produces more or fewer than asked.
    variants = [line.strip() for line in response["response"].splitlines() if line.strip()]
    return [query] + variants[:num_variants]
