"""
Input validation utilities for the Speculative RAG application.
"""
from pathlib import Path
from typing import List
from .exceptions import ValidationError
from .logger import logger


def validate_pdf_files(file_paths: List[str]) -> List[Path]:
    """
    Validate that PDF files exist and are readable.
    
    Args:
        file_paths: List of file paths to validate
        
    Returns:
        List of validated Path objects
        
    Raises:
        ValidationError: If validation fails
    """
    if not file_paths:
        raise ValidationError("No PDF files provided")
    
    validated_paths = []
    for file_path in file_paths:
        path = Path(file_path)
        
        if not path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if not path.is_file():
            raise ValidationError(f"Not a file: {file_path}")
        
        if path.suffix.lower() != '.pdf':
            raise ValidationError(f"Not a PDF file: {file_path}")
        
        if not path.stat().st_size > 0:
            logger.warning(f"Empty PDF file: {file_path}")
            continue
        
        validated_paths.append(path)
    
    if not validated_paths:
        raise ValidationError("No valid PDF files found")
    
    logger.info(f"Validated {len(validated_paths)} PDF files")
    return validated_paths


def validate_query(query: str) -> str:
    """
    Validate and sanitize user query.
    
    Args:
        query: User query string
        
    Returns:
        Sanitized query string
        
    Raises:
        ValidationError: If query is invalid
    """
    if not query or not query.strip():
        raise ValidationError("Query cannot be empty")
    
    query = query.strip()
    
    if len(query) < 3:
        raise ValidationError("Query too short (minimum 3 characters)")
    
    if len(query) > 1000:
        raise ValidationError("Query too long (maximum 1000 characters)")
    
    return query


def validate_directory(dir_path: str, create: bool = False) -> Path:
    """
    Validate that directory exists or create it.
    
    Args:
        dir_path: Directory path
        create: Whether to create directory if it doesn't exist
        
    Returns:
        Validated Path object
        
    Raises:
        ValidationError: If validation fails
    """
    path = Path(dir_path)
    
    if not path.exists():
        if create:
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                raise ValidationError(f"Failed to create directory {dir_path}: {e}")
        else:
            raise ValidationError(f"Directory does not exist: {dir_path}")
    
    if not path.is_dir():
        raise ValidationError(f"Not a directory: {dir_path}")
    
    return path
