"""
Cache management for embeddings and vector indices.
"""
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
from .logger import logger


class CacheManager:
    """Manages caching of embeddings and indices."""
    
    def __init__(self, cache_dir: str = "vector_store"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
    
    def _compute_hash(self, data: str) -> str:
        """Compute SHA256 hash of data."""
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _get_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache metadata: {e}")
    
    def get_cache_key(self, texts: list, config: dict) -> str:
        """
        Generate cache key from texts and configuration.
        
        Args:
            texts: List of text chunks
            config: Configuration dictionary
            
        Returns:
            Cache key string
        """
        # Create a deterministic string from texts and config
        text_hash = self._compute_hash("".join(sorted(texts)))
        
        # Include relevant config parameters
        config_str = json.dumps({
            'embedding_model': config.get('embedding_model'),
            'chunk_size': config.get('chunk_size'),
            'chunk_overlap': config.get('chunk_overlap')
        }, sort_keys=True)
        config_hash = self._compute_hash(config_str)
        
        return f"{text_hash[:16]}_{config_hash[:16]}"
    
    def is_cached(self, cache_key: str) -> bool:
        """
        Check if embeddings are cached for given key.
        
        Args:
            cache_key: Cache key to check
            
        Returns:
            True if cache exists and is valid
        """
        metadata = self._get_metadata()
        
        if cache_key not in metadata:
            return False
        
        cache_info = metadata[cache_key]
        index_path = Path(cache_info.get('index_path', ''))
        embeddings_path = Path(cache_info.get('embeddings_path', ''))
        
        if not (index_path.exists() and embeddings_path.exists()):
            logger.warning(f"Cache files missing for key {cache_key}")
            return False
        
        logger.info(f"Cache hit for key {cache_key}")
        return True
    
    def get_cache_paths(self, cache_key: str) -> Optional[Dict[str, str]]:
        """
        Get paths to cached files.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Dictionary with index_path and embeddings_path, or None
        """
        metadata = self._get_metadata()
        
        if cache_key not in metadata:
            return None
        
        cache_info = metadata[cache_key]
        return {
            'index_path': cache_info.get('index_path'),
            'embeddings_path': cache_info.get('embeddings_path')
        }
    
    def save_cache_info(
        self,
        cache_key: str,
        index_path: str,
        embeddings_path: str,
        num_chunks: int,
        embedding_dim: int
    ):
        """
        Save cache information.
        
        Args:
            cache_key: Cache key
            index_path: Path to FAISS index
            embeddings_path: Path to embeddings file
            num_chunks: Number of text chunks
            embedding_dim: Dimension of embeddings
        """
        metadata = self._get_metadata()
        
        metadata[cache_key] = {
            'index_path': str(index_path),
            'embeddings_path': str(embeddings_path),
            'num_chunks': num_chunks,
            'embedding_dim': embedding_dim
        }
        
        self._save_metadata(metadata)
        logger.info(f"Saved cache info for key {cache_key}")
    
    def clear_cache(self):
        """Clear all cache files and metadata."""
        try:
            if self.metadata_file.exists():
                self.metadata_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
