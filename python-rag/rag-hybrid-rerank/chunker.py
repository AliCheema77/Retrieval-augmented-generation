from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_text(filepath: str) -> str:
    """Read the full contents of a text file."""
    with open(filepath, 'r', encoding="utf-8") as f:
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping fixed-size chunks so retrieval can match
    focused passages instead of the whole document.

    The overlap exists so a sentence cut at a chunk boundary still appears
    in full inside at least one neighboring chunk, instead of losing
    context right at the cut point.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start: end]
        chunks.append(chunk)
        # Advancing by (chunk_size - overlap) instead of chunk_size is what
        # creates the overlap: the next chunk re-includes the last
        # `overlap` characters of this one.
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    sample_path = SCRIPT_DIR / "sample.txt"
    sample_text = load_text(sample_path)
    result = chunk_text(sample_text)

    print(f"Total chunks: {len(result)}")
    print("First chunk:\n", result[0])
