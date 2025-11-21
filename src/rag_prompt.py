from typing import List
from .strings import RAG_PROMPT_TEMPLATE, DOC_PREFIX, DOC_SEPARATOR


def build_rag_prompt(query: str, retrieved_docs: List[str]) -> str:
    """Build RAG prompt from query and retrieved documents."""
    context = DOC_SEPARATOR.join(
        [DOC_PREFIX.format(index=i+1, content=d) for i, d in enumerate(retrieved_docs)]
    )
    
    return RAG_PROMPT_TEMPLATE.format(context=context, query=query)
