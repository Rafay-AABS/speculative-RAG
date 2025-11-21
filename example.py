"""
Example script demonstrating programmatic usage of Speculative RAG.
"""
from dotenv import load_dotenv
from pathlib import Path

from src.config import load_config
from src.logger import setup_logger
from src.pdf_parser import parse_pdfs
from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.pipeline import SpeculativeRAG
from src.cache import CacheManager


def main():
    """Example of programmatic usage."""
    # Load environment
    load_dotenv()
    
    # Setup logger
    logger = setup_logger(log_level="INFO")
    logger.info("Starting example script")
    
    # Load configuration
    config = load_config(
        chunk_size=300,
        chunk_overlap=50,
        top_k=5
    )
    
    print("Configuration loaded successfully")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Draft Model: {config.draft_model}")
    print(f"Target Model: {config.target_model}\n")
    
    # Get PDF files
    pdf_dir = Path(config.data_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {config.data_dir}")
        print("Please add some PDF files and try again.")
        return
    
    print(f"Found {len(pdf_files)} PDF files\n")
    
    # Parse PDFs
    print("Parsing PDFs...")
    raw_text = parse_pdfs([str(f) for f in pdf_files])
    print(f"Extracted {len(raw_text)} characters\n")
    
    # Chunk text
    print("Chunking text...")
    chunks = chunk_text(raw_text, config.chunk_size, config.chunk_overlap)
    print(f"Created {len(chunks)} chunks\n")
    
    # Initialize cache manager
    cache_manager = CacheManager(config.vector_store_dir)
    cache_key = cache_manager.get_cache_key(chunks, config.to_dict())
    
    # Build or load embeddings
    if cache_manager.is_cached(cache_key):
        print("Using cached embeddings\n")
        cache_paths = cache_manager.get_cache_paths(cache_key)
        retriever = Retriever(
            index_path=cache_paths['index_path'],
            emb_path=cache_paths['embeddings_path']
        )
    else:
        print("Building new embeddings...")
        embedder = Embedder(config.embedding_model)
        embeddings = embedder.embed_texts(chunks)
        embedder.build_faiss(embeddings, config.vector_store_dir)
        
        # Save cache info
        index_path = Path(config.vector_store_dir) / "index.faiss"
        emb_path = Path(config.vector_store_dir) / "embeddings.npy"
        cache_manager.save_cache_info(
            cache_key,
            str(index_path),
            str(emb_path),
            len(chunks),
            embeddings.shape[1]
        )
        
        retriever = Retriever()
        print()
    
    # Create RAG pipeline
    print("Initializing RAG pipeline...")
    pipeline = SpeculativeRAG(retriever)
    print("Pipeline ready!\n")
    
    # Run example queries
    queries = [
        "What are the main topics covered in the documents?",
        "Can you summarize the key points?",
        "What recommendations are provided?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        try:
            answer = pipeline.run(query, chunks)
            print(f"\nAnswer:\n{answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
