from pathlib import Path

SCRIPT_DIR = Path(__file__).parent


def load_text(filepath: str) -> str:
    with open(filepath, 'r', encoding="utf-8") as f:
        return f.read()
    

def chunk_text(text: str, chunk_size: int=500, overlap: int=50) -> list[str]:
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start: end]
        chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks
