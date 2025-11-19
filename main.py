# main.py

from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.pipeline import SpeculativeRAG
import glob

# 1. Load Raw Textt
files = glob.glob("data/raw/*.txt")
raw_text = "\n\n".join(open(f).read() for f in files)

# 2. Chunk
chunks = chunk_text(raw_text)

# 3. Build embeddings + vector store
embed = Embedder()
emb = embed.embed_texts(chunks)
embed.build_faiss(emb)

# 4. Create retriever
retriever = Retriever()

# 5. Create RAG pipeline
pipeline = SpeculativeRAG(retriever)

# 6. Run query
query = "Explain the leadership principles in these documents."
answer = pipeline.run(query, chunks)

print("\n--- Answer ---\n")
print(answer)