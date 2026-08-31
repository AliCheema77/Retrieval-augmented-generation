from pathlib import Path        # Path: object-oriented filesystem paths (built-in pathlib)

# __file__ is the path to this script; .parent gives its containing directory.
# Used so file paths (like sample.txt) work regardless of where the script is run from.
SCRIPT_DIR = Path(__file__).parent


def load_documents(directory) -> list[tuple[str, str]]:
    """Load every .txt file in a directory, returning (filename, text) pairs."""
    documents = []
    for filepath in sorted(Path(directory).glob("*.txt")):
        documents.append((filepath.name, load_text(filepath)))
    return documents


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


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 70) -> list[str]:
    """
    Split text into chunks that respect line boundaries instead of cutting
    mid-line, packing consecutive lines together up to chunk_size and only
    falling back to a hard character cut if a single line itself exceeds
    chunk_size.
    """
    lines = text.split("\n")
    chunks = []
    current_lines = []
    current_length = 0

    for line in lines:
        line_length = len(line) + 1  # +1 for the newline that rejoins it

        # A single line longer than chunk_size can't be packed with anything
        # else — flush what we have, then hard-cut this one line as a fallback.
        if line_length > chunk_size:
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_length = 0
            for start in range(0, len(line), chunk_size):
                chunks.append(line[start:start + chunk_size])
            continue

        # Adding this line would overflow the current chunk — close it out,
        # then carry the last few lines forward into the next chunk as overlap.
        if current_length + line_length > chunk_size and current_lines:
            chunks.append("\n".join(current_lines))

            overlap_lines = []
            overlap_length = 0
            for prev_line in reversed(current_lines):
                if overlap_length + len(prev_line) + 1 > overlap:
                    break
                overlap_lines.insert(0, prev_line)
                overlap_length += len(prev_line) + 1

            current_lines = overlap_lines
            current_length = overlap_length

        current_lines.append(line)
        current_length += line_length

    if current_lines:
        chunks.append("\n".join(current_lines))

    return chunks
