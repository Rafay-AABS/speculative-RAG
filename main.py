from dotenv import load_dotenv
import os
from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.pipeline import SpeculativeRAG
from src.strings import (
    DATA_RAW_PATTERN,
    ENV_HF_TOKEN,
    ENV_HUGGING_FACE_HUB_TOKEN,
    ERROR_NO_TEXT_FILES,
    ERROR_EMPTY_FILES,
    ERROR_NO_CHUNKS,
    DEFAULT_QUERY,
    ANSWER_HEADER,
    DOC_SEPARATOR
)
import glob

# Load environment variables from .env file
load_dotenv()

# Set HuggingFace token for gated models
if os.getenv(ENV_HF_TOKEN):
    os.environ[ENV_HF_TOKEN] = os.getenv(ENV_HF_TOKEN)
    os.environ[ENV_HUGGING_FACE_HUB_TOKEN] = os.getenv(ENV_HF_TOKEN)

# 1. Load Raw Text
files = glob.glob(DATA_RAW_PATTERN)
if not files:
    raise ValueError(ERROR_NO_TEXT_FILES)

raw_text = DOC_SEPARATOR.join(open(f).read() for f in files)

if not raw_text.strip():
    raise ValueError(ERROR_EMPTY_FILES)

# 2. Chunk
chunks = chunk_text(raw_text)
if not chunks:
    raise ValueError(ERROR_NO_CHUNKS)

# 3. Build embeddings + vector store
embed = Embedder()
emb = embed.embed_texts(chunks)
embed.build_faiss(emb)

# 4. Create retriever
retriever = Retriever()

# 5. Create RAG pipeline
pipeline = SpeculativeRAG(retriever)

# 6. Run query
query = DEFAULT_QUERY
answer = pipeline.run(query, chunks)

print(ANSWER_HEADER)
print(answer)