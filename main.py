from dotenv import load_dotenv
import os
from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.pipeline import SpeculativeRAG
import glob

# Load environment variables from .env file
load_dotenv()

# Set HuggingFace token for gated models
if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.getenv("HF_TOKEN")

# 1. Load Raw Text
files = glob.glob("data/raw/*.txt")
if not files:
    raise ValueError("No text files found in data/raw/. Please add some .txt files to process.")

raw_text = "\n\n".join(open(f).read() for f in files)

if not raw_text.strip():
    raise ValueError("All text files are empty. Please add content to process.")

# 2. Chunk
chunks = chunk_text(raw_text)
if not chunks:
    raise ValueError("No chunks were created from the text. Please check your data.")

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