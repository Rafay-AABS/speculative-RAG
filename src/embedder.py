from typing import List, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from .strings import (
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_DIR,
    ERROR_EMPTY_TEXT_LIST,
    ERROR_INVALID_EMBEDDINGS,
    SUCCESS_FAISS_SAVED
)
from .exceptions import EmbeddingError
from .logger import logger


class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        logger.info(f"Initializing embedder with model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise EmbeddingError(f"Failed to load embedding model {model_name}: {e}") from e

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise EmbeddingError(ERROR_EMPTY_TEXT_LIST)
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    def build_faiss(self, embeddings: np.ndarray, save_dir: str = VECTOR_STORE_DIR):
        if embeddings.size == 0 or len(embeddings.shape) != 2:
            raise EmbeddingError(ERROR_INVALID_EMBEDDINGS.format(shape=embeddings.shape))
        
        try:
            dim = embeddings.shape[1]
            logger.info(f"Building FAISS index with dimension: {dim}")
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

            os.makedirs(save_dir, exist_ok=True)

            index_path = f"{save_dir}/index.faiss"
            emb_path = f"{save_dir}/embeddings.npy"
            
            faiss.write_index(index, index_path)
            np.save(emb_path, embeddings)

            logger.info(SUCCESS_FAISS_SAVED)
            print(SUCCESS_FAISS_SAVED)
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            raise EmbeddingError(f"Failed to build FAISS index: {e}") from e

    def load_faiss(self, save_dir: str = VECTOR_STORE_DIR) -> Tuple[faiss.Index, np.ndarray]:
        try:
            logger.info(f"Loading FAISS index from {save_dir}")
            index = faiss.read_index(f"{save_dir}/index.faiss")
            embeddings = np.load(f"{save_dir}/embeddings.npy")
            logger.info(f"Loaded index with {index.ntotal} vectors")
            return index, embeddings
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise EmbeddingError(f"Failed to load FAISS index: {e}") from e
