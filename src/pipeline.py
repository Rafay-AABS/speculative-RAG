from typing import List
from src.rag_prompt import build_rag_prompt
from src.speculative_decoder import speculative_decode
from src.logger import logger
from models.draft_model import DraftModel
from models.target_model import TargetModel


class SpeculativeRAG:
    def __init__(self, retriever):
        logger.info("Initializing SpeculativeRAG pipeline")
        self.retriever = retriever
        self.draft = DraftModel()
        self.target = TargetModel()
        logger.info("Pipeline initialized successfully")

    def run(self, query: str, texts: List[str]) -> str:
        """Run the complete RAG pipeline."""
        logger.info(f"Running pipeline for query: {query[:100]}...")
        
        # Retrieve relevant documents
        docs = self.retriever.retrieve(query, texts)
        logger.debug(f"Retrieved {len(docs)} documents")
        
        # Build prompt
        prompt = build_rag_prompt(query, docs)
        logger.debug("Built RAG prompt")
        
        # Generate answer using speculative decoding
        answer = speculative_decode(prompt, self.draft, self.target)
        logger.info("Generated answer successfully")
        
        return answer
