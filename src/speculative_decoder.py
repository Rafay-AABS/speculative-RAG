def speculative_decode(prompt, draft_model, target_model):
    """
    Speculative decoding with API-based models.
    Draft model generates quickly, target model verifies/refines.
    """
    draft_text = draft_model.generate(prompt)
    final_text = target_model.verify(prompt, draft_text)
    
    return final_text