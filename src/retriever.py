from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from .strings import (
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K_RESULTS
)
from .exceptions import RetrievalError
from .logger import logger


class Retriever:
    def __init__(self, index_path: str = FAISS_INDEX_PATH, emb_path: str = EMBEDDINGS_PATH):
        try:
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            self.embeddings = np.load(emb_path)
            self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(f"Retriever initialized with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {e}")
            raise RetrievalError(f"Failed to initialize retriever: {e}") from e

    def retrieve(self, query: str, texts: List[str], k: int = TOP_K_RESULTS) -> List[str]:
        try:
            logger.debug(f"Retrieving top-{k} documents for query: {query[:50]}...")
            q_emb = self.model.encode([query])
            distances, ids = self.index.search(q_emb, k)
            
            retrieved = [texts[i] for i in ids[0] if i < len(texts)]
            logger.info(f"Retrieved {len(retrieved)} documents")
            return retrieved
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            raise RetrievalError(f"Failed to retrieve documents: {e}") from e
