from typing import List
from .strings import CHUNK_SIZE, CHUNK_OVERLAP
from .logger import logger


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Chunks data into configurable word chunks with overlap"""
    logger.debug(f"Chunking text with size={chunk_size}, overlap={overlap}")
    
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    logger.info(f"Created {len(chunks)} chunks from text")
    return chunks
