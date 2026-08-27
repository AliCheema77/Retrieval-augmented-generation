from pathlib import Path        # Path: object-oriented filesystem paths (built-in pathlib)

# __file__ is the path to this script; .parent gives its containing directory.
# Used so file paths (like sample.txt) work regardless of where the script is run from.
SCRIPT_DIR = Path(__file__).parent


def load_text(filepath: str) -> str:
    """
    Read the entire contents of a text file into memory as a single string.

    Parameters:
        filepath: path (str or Path) to the text file to read.

    Returns:
        The full file contents as one string.
    """
    # open(): built-in function that opens a file and returns a file handle.
    #   'r'          -> open in read (text) mode
    #   encoding="utf-8" -> decode bytes as UTF-8 text (avoids errors with
    #                       non-ASCII characters like em-dashes, accents, etc.)
    # The "with" statement (context manager) guarantees the file is closed
    # automatically once the block exits, even if an error occurs.
    with open(filepath, 'r', encoding="utf-8") as f:
        # f.read(): built-in file-object method that reads the whole file
        # at once and returns it as a single string.
        return f.read()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split a long piece of text into smaller, overlapping fixed-size chunks.

    This exists because embedding/retrieval models work best on short,
    focused passages rather than an entire document — chunking lets the
    retriever later match a query against a specific paragraph-sized piece
    instead of the whole text at once.

    Parameters:
        text: the full text to split.
        chunk_size: number of characters per chunk.
        overlap: number of characters shared between consecutive chunks,
            so a sentence that gets cut at a chunk boundary still appears
            in full inside at least one neighboring chunk (prevents losing
            context right at the cut point).

    Returns:
        A list of chunk strings, in order, covering the whole input text.
    """
    chunks = []
    start = 0
    # len(): built-in function returning the number of characters in the string.
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        # Python string slicing: text[start:end] extracts characters from
        # index `start` up to (but not including) index `end`. If `end`
        # exceeds the string length, Python just slices to the end safely.
        chunk = text[start: end]
        chunks.append(chunk)
        # Advance the window by (chunk_size - overlap) instead of chunk_size,
        # so the next chunk re-includes the last `overlap` characters of
        # this one — that's what creates the overlap between chunks.
        start += chunk_size - overlap

    return chunks


# This block only runs when the script is executed directly
# (e.g. `python chunker.py`), not when imported as a module elsewhere.
if __name__ == "__main__":
    # Build the full path to the sample document living next to this script.
    sample_path = SCRIPT_DIR / "sample.txt"

    # Read the raw file contents into a single string.
    sample_text = load_text(sample_path)

    # Split that string into overlapping chunks using the default sizes.
    result = chunk_text(sample_text)

    # print(): built-in function that writes text to standard output (the terminal).
    # len(result) -> how many chunks were produced.
    print(f"Total chunks: {len(result)}")
    print("First chunk:\n", result[0])
