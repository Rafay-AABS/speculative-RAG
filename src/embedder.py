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

class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        if not texts:
            raise ValueError(ERROR_EMPTY_TEXT_LIST)
        return self.model.encode(texts, convert_to_numpy=True)

    def build_faiss(self, embeddings, save_dir=VECTOR_STORE_DIR):
        if embeddings.size == 0 or len(embeddings.shape) != 2:
            raise ValueError(ERROR_INVALID_EMBEDDINGS.format(shape=embeddings.shape))
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        os.makedirs(save_dir, exist_ok=True)

        faiss.write_index(index, f"{save_dir}/index.faiss")
        np.save(f"{save_dir}/embeddings.npy", embeddings)

        print(SUCCESS_FAISS_SAVED)

    def load_faiss(self, save_dir=VECTOR_STORE_DIR):
        index = faiss.read_index(f"{save_dir}/index.faiss")
        embeddings = np.load(f"{save_dir}/embeddings.npy")
        return index, embeddings
