from .strings import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Chunks data into configurable word chunks with overlap"""

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
