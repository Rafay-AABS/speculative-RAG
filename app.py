from typing import List, Optional
from pathlib import Path
import tempfile
import shutil
import os

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel

from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.pipeline import SpeculativeRAG
from src.document_parser import parse_documents
from src.config import load_config, Config
from src.cache import CacheManager
from src.logger import setup_logger, logger
from src.exceptions import DocumentParsingError
from src.strings import ENV_HF_TOKEN, ENV_HUGGING_FACE_HUB_TOKEN


app = FastAPI(
    title="Speculative RAG API",
    description="Fast RAG over uploaded PDFs",
    version="1.0.0",
)


class AppState:
    def __init__(self):
        self.config: Optional[Config] = None
        self.chunks: Optional[List[str]] = None
        self.retriever: Optional[Retriever] = None
        self.pipeline: Optional[SpeculativeRAG] = None
        self.cache_manager: Optional[CacheManager] = None
        self.initialized: bool = False


state = AppState()


def initialize_app() -> None:
    if state.initialized:
        return

    load_dotenv()
    setup_logger()

    state.config = load_config()

    # HuggingFace token if present
    if state.config.hf_token:
        os.environ[ENV_HF_TOKEN] = state.config.hf_token
        os.environ[ENV_HUGGING_FACE_HUB_TOKEN] = state.config.hf_token
        logger.debug("HuggingFace token configured")

    state.cache_manager = CacheManager(
        cache_dir=state.config.vector_store_dir,
    )

    state.initialized = True
    logger.info("FastAPI app initialized")


@app.on_event("startup")
async def on_startup() -> None:
    initialize_app()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    answer: str


class UploadResponse(BaseModel):
    message: str
    num_files: int
    num_chunks: int


@app.get("/")
async def root():
    return {
        "message": "Speculative RAG API",
        "status": "running",
        "endpoints": {"upload": "/upload-documents", "query": "/query"},
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "documents_loaded": state.chunks is not None,
        "pipeline_ready": state.pipeline is not None,
    }


@app.post("/upload-documents", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(..., description="Document files to process (PDF or Word)"),
    force_rebuild: bool = Form(False, description="Force rebuild of embeddings"),
):
    """Endpoint to upload and process one or more document files (PDF or Word)."""
    initialize_app()

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    supported_extensions = [".pdf", ".docx", ".doc"]
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} has unsupported format. Supported formats: {', '.join(supported_extensions)}",
            )

    logger.info("Received %d document files for processing", len(files))

    temp_dir = tempfile.mkdtemp()
    document_paths: List[str] = []

    try:
        # Save uploaded files to a temporary directory
        for file in files:
            temp_path = Path(temp_dir) / file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            document_paths.append(str(temp_path))

        # Parse documents to text
        try:
            raw_text = parse_documents(document_paths)
        except DocumentParsingError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text content extracted from documents",
            )

        # Chunk text
        state.chunks = chunk_text(
            raw_text,
            state.config.chunk_size,
            state.config.chunk_overlap,
        )

        if not state.chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks created from documents",
            )

        # Build / load embeddings + index using cache
        cache_key = state.cache_manager.get_cache_key(
            state.chunks,
            state.config.to_dict(),
        )
        use_cache = not force_rebuild and state.cache_manager.is_cached(cache_key)

        if use_cache:
            cache_paths = state.cache_manager.get_cache_paths(cache_key)
            state.retriever = Retriever(
                index_path=cache_paths["index_path"],
                emb_path=cache_paths["embeddings_path"],
            )
        else:
            embedder = Embedder(state.config.embedding_model)
            embeddings = embedder.embed_texts(state.chunks)

            state.retriever = Retriever()
            state.retriever.build_index(embeddings)

            state.cache_manager.save_cache(
                cache_key,
                state.retriever,
                embeddings,
            )

        # Initialize RAG pipeline
        state.pipeline = SpeculativeRAG(state.retriever)

        logger.info("Documents processed successfully: %d chunks", len(state.chunks))

        return UploadResponse(
            message="Documents processed successfully",
            num_files=len(files),
            num_chunks=len(state.chunks),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Backward-compatible alias for PDF uploads
@app.post("/upload-pdf", response_model=UploadResponse)
async def upload_pdf(
    files: List[UploadFile] = File(..., description="PDF files to process"),
    force_rebuild: bool = Form(False, description="Force rebuild of embeddings"),
):
    """Backward-compatible endpoint for PDF uploads. Redirects to /upload-documents."""
    return await upload_documents(files, force_rebuild)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Endpoint to query the processed documents using RAG."""
    initialize_app()

    if state.chunks is None or state.pipeline is None:
        raise HTTPException(
            status_code=400,
            detail="No documents loaded. Please upload documents first using /upload-documents.",
        )

    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info("Processing query: %s", query_text[:100])

    try:
        answer = state.pipeline.run(query_text, state.chunks)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Error while running RAG pipeline: %s", str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing query") from exc

    return QueryResponse(query=query_text, answer=answer)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)