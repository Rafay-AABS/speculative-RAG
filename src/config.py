"""
Configuration management for the Speculative RAG application.
"""
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
from .strings import (
    EMBEDDING_MODEL_NAME,
    DRAFT_MODEL_NAME,
    TARGET_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    DRAFT_MAX_TOKENS,
    DRAFT_TEMPERATURE,
    TARGET_MAX_TOKENS,
    TARGET_TEMPERATURE,
    VECTOR_STORE_DIR,
    DATA_RAW_DIR
)


@dataclass
class Config:
    """Application configuration with validation."""
    
    # Paths
    data_dir: str = DATA_RAW_DIR
    vector_store_dir: str = VECTOR_STORE_DIR
    
    # Model configuration
    embedding_model: str = EMBEDDING_MODEL_NAME
    draft_model: str = DRAFT_MODEL_NAME
    target_model: str = TARGET_MODEL_NAME
    
    # Chunking parameters
    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    
    # Retrieval parameters
    top_k: int = TOP_K_RESULTS
    
    # Generation parameters
    draft_max_tokens: int = DRAFT_MAX_TOKENS
    draft_temperature: float = DRAFT_TEMPERATURE
    target_max_tokens: int = TARGET_MAX_TOKENS
    target_temperature: float = TARGET_TEMPERATURE
    
    # Runtime settings
    log_level: str = "INFO"
    interactive_mode: bool = False
    rebuild_index: bool = False
    
    # API keys (loaded from environment)
    groq_api_key: Optional[str] = field(default=None, repr=False)
    hf_token: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Validate configuration and load environment variables."""
        self._load_env_vars()
        self._validate()
    
    def _load_env_vars(self):
        """Load API keys from environment variables."""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
    
    def _validate(self):
        """Validate configuration values."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive, got {self.top_k}")
        
        if not 0 <= self.draft_temperature <= 2:
            raise ValueError(f"draft_temperature must be between 0 and 2, got {self.draft_temperature}")
        
        if not 0 <= self.target_temperature <= 2:
            raise ValueError(f"target_temperature must be between 0 and 2, got {self.target_temperature}")
        
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        # Create directories if they don't exist
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (excluding sensitive data)."""
        data = self.__dict__.copy()
        data.pop('groq_api_key', None)
        data.pop('hf_token', None)
        return data


def load_config(**kwargs) -> Config:
    """
    Load configuration with optional overrides.
    
    Args:
        **kwargs: Configuration overrides
        
    Returns:
        Config instance
    """
    return Config(**kwargs)
