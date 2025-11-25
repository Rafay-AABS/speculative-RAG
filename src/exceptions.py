"""
Custom exceptions for the Speculative RAG application.
"""


class SpeculativeRAGError(Exception):
    """Base exception for all Speculative RAG errors."""
    pass


class ConfigurationError(SpeculativeRAGError):
    """Raised when configuration is invalid."""
    pass


class PDFParsingError(SpeculativeRAGError):
    """Raised when PDF parsing fails."""
    pass


class DocumentParsingError(SpeculativeRAGError):
    """Raised when document parsing fails (PDF, Word, etc.)."""
    pass


class EmbeddingError(SpeculativeRAGError):
    """Raised when embedding generation or indexing fails."""
    pass


class RetrievalError(SpeculativeRAGError):
    """Raised when document retrieval fails."""
    pass


class ModelError(SpeculativeRAGError):
    """Raised when model inference fails."""
    pass


class ValidationError(SpeculativeRAGError):
    """Raised when input validation fails."""
    pass
