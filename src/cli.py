"""
Command-line interface for the Speculative RAG application.
"""
import argparse
from pathlib import Path
from .strings import (
    EMBEDDING_MODEL_NAME,
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    VECTOR_STORE_DIR,
    DATA_RAW_DIR
)


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Speculative RAG: Fast RAG system using speculative decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with PDFs from default directory
  python main.py --interactive
  
  # Single query with specific PDFs
  python main.py --query "What are the key points?" --pdf file1.pdf file2.pdf
  
  # Rebuild index and use custom parameters
  python main.py --query "Summarize" --rebuild --top-k 10 --chunk-size 500
  
  # Use custom models
  python main.py --query "Explain" --draft-model llama-3.1-8b-instant
        """
    )
    
    # Input options
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument(
        '-q', '--query',
        type=str,
        help='Query string to process (if not provided, enters interactive mode)'
    )
    input_group.add_argument(
        '--pdf',
        nargs='+',
        type=str,
        metavar='FILE',
        help='PDF files to process (if not provided, uses all PDFs in data directory)'
    )
    input_group.add_argument(
        '--data-dir',
        type=str,
        default=DATA_RAW_DIR,
        metavar='DIR',
        help=f'Directory containing PDF files (default: {DATA_RAW_DIR})'
    )
    
    # Mode options
    mode_group = parser.add_argument_group('Mode Options')
    mode_group.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode for multiple queries'
    )
    mode_group.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild of vector index (default: use existing if available)'
    )
    
    # Model configuration
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--embedding-model',
        type=str,
        default=EMBEDDING_MODEL_NAME,
        metavar='MODEL',
        help=f'Embedding model name (default: {EMBEDDING_MODEL_NAME})'
    )
    model_group.add_argument(
        '--draft-model',
        type=str,
        default=DRAFT_MODEL_NAME,
        metavar='MODEL',
        help=f'Draft model name (default: {DRAFT_MODEL_NAME})'
    )
    model_group.add_argument(
        '--target-model',
        type=str,
        default=TARGET_MODEL_NAME,
        metavar='MODEL',
        help=f'Target model name (default: {TARGET_MODEL_NAME})'
    )
    
    # RAG parameters
    rag_group = parser.add_argument_group('RAG Parameters')
    rag_group.add_argument(
        '--chunk-size',
        type=int,
        default=CHUNK_SIZE,
        metavar='N',
        help=f'Number of words per chunk (default: {CHUNK_SIZE})'
    )
    rag_group.add_argument(
        '--chunk-overlap',
        type=int,
        default=CHUNK_OVERLAP,
        metavar='N',
        help=f'Number of overlapping words between chunks (default: {CHUNK_OVERLAP})'
    )
    rag_group.add_argument(
        '--top-k',
        type=int,
        default=TOP_K_RESULTS,
        metavar='N',
        help=f'Number of documents to retrieve (default: {TOP_K_RESULTS})'
    )
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--vector-store',
        type=str,
        default=VECTOR_STORE_DIR,
        metavar='DIR',
        help=f'Directory for vector store (default: {VECTOR_STORE_DIR})'
    )
    output_group.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    output_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output (equivalent to --log-level DEBUG)'
    )
    
    args = parser.parse_args()
    
    # Adjust log level for verbose mode
    if args.verbose:
        args.log_level = 'DEBUG'
    
    # If no query provided and not interactive, enable interactive mode
    if not args.query and not args.interactive:
        args.interactive = True
    
    return args


def display_config(config):
    """Display current configuration to user."""
    print("\n" + "="*60)
    print("Speculative RAG Configuration")
    print("="*60)
    print(f"Embedding Model:    {config.embedding_model}")
    print(f"Draft Model:        {config.draft_model}")
    print(f"Target Model:       {config.target_model}")
    print(f"Chunk Size:         {config.chunk_size} words")
    print(f"Chunk Overlap:      {config.chunk_overlap} words")
    print(f"Top-K Results:      {config.top_k}")
    print(f"Vector Store:       {config.vector_store_dir}")
    print(f"Data Directory:     {config.data_dir}")
    print(f"Interactive Mode:   {config.interactive_mode}")
    print(f"Rebuild Index:      {config.rebuild_index}")
    print("="*60 + "\n")
