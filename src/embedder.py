import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class Embedder:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)

    def build_faiss(self, embeddings, save_dir="vector_store"):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        os.makedirs(save_dir, exist_ok=True)

        faiss.write_index(index, f"{save_dir}/index.faiss")
        np.save(f"{save_dir}/embeddings.npy", embeddings)

        print("Saved FAISS index + embeddings.")

    def load_faiss(self, save_dir="vector_store"):
        index = faiss.read_index(f"{save_dir}/index.faiss")
        embeddings = np.load(f"{save_dir}/embeddings.npy")
        return index, embeddings
