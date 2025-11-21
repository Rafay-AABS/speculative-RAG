from src.logger import logger


def speculative_decode(prompt: str, draft_model, target_model) -> str:
    """Execute speculative decoding strategy."""
    logger.debug("Starting speculative decoding")
    
    draft_text = draft_model.generate(prompt)
    final_text = target_model.verify(prompt, draft_text)
    
    logger.info("Speculative decoding completed")
    return final_text