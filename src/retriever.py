
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class Retriever:
    def __init__(self, index_path="vector_store/index.faiss", emb_path="vector_store/embeddings.npy"):
        self.index = faiss.read_index(index_path)
        self.embeddings = np.load(emb_path)
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def retrieve(self, query, texts, k=5):
        q_emb = self.model.encode([query])
        distances, ids = self.index.search(q_emb, k)
        return [texts[i] for i in ids[0]]
