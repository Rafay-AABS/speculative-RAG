"""
Centralized string constants and configuration values for the Speculative RAG project.
"""

# File paths
DATA_RAW_DIR = "data/raw"
DATA_RAW_PATTERN = "data/raw/*.txt"
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_PATH = "vector_store/index.faiss"
EMBEDDINGS_PATH = "vector_store/embeddings.npy"

# Model names
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DRAFT_MODEL_NAME = "llama-3.1-8b-instant"
TARGET_MODEL_NAME = "llama-3.3-70b-versatile"

# Environment variables
ENV_HF_TOKEN = "HF_TOKEN"
ENV_HUGGING_FACE_HUB_TOKEN = "HUGGING_FACE_HUB_TOKEN"
ENV_GROQ_API_KEY = "GROQ_API_KEY"

# Chunking parameters
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Retrieval parameters
TOP_K_RESULTS = 5

# Model generation parameters
DRAFT_MAX_TOKENS = 512
DRAFT_TEMPERATURE = 0.7
TARGET_MAX_TOKENS = 1024
TARGET_TEMPERATURE = 0.3

# User messages
USER_ROLE = "user"

# Prompts and messages
RAG_PROMPT_TEMPLATE = """
        You are a helpful assistant.

        Context:
        {context}

        Question: {query}
        Answer:
    """
DEFAULT_QUERY = "Explain the leadership principles in these documents."

# Error messages
ERROR_NO_TEXT_FILES = "No text files found in data/raw/. Please add some .txt files to process."
ERROR_EMPTY_FILES = "All text files are empty. Please add content to process."
ERROR_NO_CHUNKS = "No chunks were created from the text. Please check your data."
ERROR_EMPTY_TEXT_LIST = "Cannot embed empty text list"
ERROR_INVALID_EMBEDDINGS = "Invalid embeddings shape: {shape}. Expected 2D array with shape (n_samples, n_features)"
ERROR_GROQ_API_KEY = "GROQ_API_KEY not found. Get free API key from https://console.groq.com"

# Success messages
SUCCESS_FAISS_SAVED = "Saved FAISS index + embeddings."

# Output formatting
ANSWER_HEADER = "\n--- Answer ---\n"
DOC_PREFIX = "[Doc {index}]\n{content}"
DOC_SEPARATOR = "\n\n"
