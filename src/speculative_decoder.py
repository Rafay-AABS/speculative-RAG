def speculative_decode(prompt, draft_model, target_model):
    
    draft_text = draft_model.generate(prompt)
    final_text = target_model.verify(prompt, draft_text)
    
    return final_text