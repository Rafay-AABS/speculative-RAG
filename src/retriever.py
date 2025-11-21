
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from .strings import (
    FAISS_INDEX_PATH,
    EMBEDDINGS_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K_RESULTS
)

class Retriever:
    def __init__(self, index_path=FAISS_INDEX_PATH, emb_path=EMBEDDINGS_PATH):
        self.index = faiss.read_index(index_path)
        self.embeddings = np.load(emb_path)
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrieve(self, query, texts, k=TOP_K_RESULTS):
        q_emb = self.model.encode([query])
        distances, ids = self.index.search(q_emb, k)
        return [texts[i] for i in ids[0]]
