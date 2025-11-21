"""
Speculative RAG: Fast RAG system using speculative decoding.

This application processes PDF documents, builds a vector index, and answers
queries using a speculative decoding approach for improved performance.
"""
from dotenv import load_dotenv
import os
import sys
import glob
from pathlib import Path
from typing import List, Optional

from src.chunker import chunk_text
from src.embedder import Embedder
from src.retriever import Retriever
from src.pipeline import SpeculativeRAG
from src.pdf_parser import parse_pdfs
from src.config import load_config, Config
from src.cli import parse_arguments, display_config
from src.logger import setup_logger, logger
from src.validators import validate_pdf_files, validate_query, validate_directory
from src.cache import CacheManager
from src.exceptions import (
    SpeculativeRAGError,
    ConfigurationError,
    PDFParsingError,
    ValidationError
)
from src.strings import (
    DATA_RAW_PATTERN,
    ENV_HF_TOKEN,
    ENV_HUGGING_FACE_HUB_TOKEN,
    ANSWER_HEADER
)


def setup_environment(config: Config):
    """Set up environment variables and directories."""
    load_dotenv()
    
    # Set HuggingFace token for gated models
    if config.hf_token:
        os.environ[ENV_HF_TOKEN] = config.hf_token
        os.environ[ENV_HUGGING_FACE_HUB_TOKEN] = config.hf_token
        logger.debug("HuggingFace token configured")


def get_pdf_files(args) -> List[str]:
    """Get list of PDF files to process."""
    if args.pdf:
        # Use specific PDF files provided
        logger.info(f"Using {len(args.pdf)} PDF files provided via command line")
        return args.pdf
    else:
        # Use all PDFs from data directory
        pattern = str(Path(args.data_dir) / "*.pdf")
        files = glob.glob(pattern)
        logger.info(f"Found {len(files)} PDF files in {args.data_dir}")
        return files


def process_documents(config: Config, pdf_files: List[str], cache_manager: CacheManager, force_rebuild: bool = False):
    """
    Process PDF documents and build/load vector index.
    
    Returns:
        tuple: (chunks, retriever)
    """
    logger.info("Processing documents...")
    
    # Parse PDFs
    raw_text = parse_pdfs(pdf_files)
    logger.info(f"Extracted text: {len(raw_text)} characters")
    
    if not raw_text.strip():
        raise PDFParsingError("No text content extracted from PDFs")
    
    # Chunk text
    chunks = chunk_text(raw_text, config.chunk_size, config.chunk_overlap)
    logger.info(f"Created {len(chunks)} text chunks")
    
    if not chunks:
        raise ValueError("No chunks created from text")
    
    # Check cache
    cache_key = cache_manager.get_cache_key(chunks, config.to_dict())
    use_cache = not force_rebuild and cache_manager.is_cached(cache_key)
    
    if use_cache:
        logger.info("Using cached embeddings")
        cache_paths = cache_manager.get_cache_paths(cache_key)
        retriever = Retriever(
            index_path=cache_paths['index_path'],
            emb_path=cache_paths['embeddings_path']
        )
    else:
        logger.info("Building new embeddings and vector index...")
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
    
    return chunks, retriever


def run_single_query(query: str, pipeline: SpeculativeRAG, chunks: List[str]):
    """Run a single query and display results."""
    try:
        query = validate_query(query)
        logger.info(f"Processing query: {query}")
        
        answer = pipeline.run(query, chunks)
        
        print(ANSWER_HEADER)
        print(answer)
        print()
        
    except ValidationError as e:
        logger.error(f"Invalid query: {e}")
        print(f"Error: {e}")
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        print(f"Error processing query: {e}")


def run_interactive_mode(pipeline: SpeculativeRAG, chunks: List[str]):
    """Run interactive query loop."""
    print("\n" + "="*60)
    print("Interactive Mode - Enter your queries (type 'quit' or 'exit' to stop)")
    print("="*60 + "\n")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting interactive mode...")
                break
            
            if not query:
                continue
            
            run_single_query(query, pipeline, chunks)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            break
        except EOFError:
            print("\n\nEnd of input. Exiting...")
            break


def main():
    """Main entry point for the application."""
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Setup logger with specified level
        global logger
        logger = setup_logger(log_level=args.log_level)
        logger.info("Starting Speculative RAG application")
        
        # Load configuration from arguments
        config = load_config(
            data_dir=args.data_dir,
            vector_store_dir=args.vector_store,
            embedding_model=args.embedding_model,
            draft_model=args.draft_model,
            target_model=args.target_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            log_level=args.log_level,
            interactive_mode=args.interactive,
            rebuild_index=args.rebuild
        )
        
        # Display configuration
        display_config(config)
        
        # Setup environment
        setup_environment(config)
        
        # Get PDF files to process
        pdf_files = get_pdf_files(args)
        
        if not pdf_files:
            raise ValidationError(f"No PDF files found in {args.data_dir}")
        
        # Validate PDF files
        validated_paths = validate_pdf_files(pdf_files)
        pdf_files = [str(p) for p in validated_paths]
        
        # Initialize cache manager
        cache_manager = CacheManager(config.vector_store_dir)
        
        # Process documents and build index
        chunks, retriever = process_documents(
            config,
            pdf_files,
            cache_manager,
            force_rebuild=config.rebuild_index
        )
        
        # Create RAG pipeline
        logger.info("Initializing RAG pipeline...")
        pipeline = SpeculativeRAG(retriever)
        logger.info("Pipeline ready")
        
        # Run queries
        if config.interactive_mode:
            run_interactive_mode(pipeline, chunks)
        else:
            run_single_query(args.query, pipeline, chunks)
        
        logger.info("Application completed successfully")
        return 0
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nConfiguration Error: {e}", file=sys.stderr)
        return 1
    
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        print(f"\nValidation Error: {e}", file=sys.stderr)
        return 1
    
    except PDFParsingError as e:
        logger.error(f"PDF parsing error: {e}")
        print(f"\nPDF Parsing Error: {e}", file=sys.stderr)
        return 1
    
    except SpeculativeRAGError as e:
        logger.error(f"Application error: {e}")
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        print("\n\nInterrupted by user. Exiting...")
        return 130
    
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        print(f"\nUnexpected Error: {e}", file=sys.stderr)
        print("Please check the log file for more details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())