def build_rag_prompt(query, retrieved_docs):
    context = "\n\n".join([f"[Doc {i+1}]\n{d}" for i, d in enumerate(retrieved_docs)])
    
    return f"""
        You are a helpful assistant.

        Context:
        {context}

        Question: {query}
        Answer:
    """
