def speculative_decode(prompt, draft_model, target_model):
    draft_tokens = draft_model.generate(prompt)
    final_tokens = target_model.verify(prompt, draft_tokens)

    return target_model.tokenizer.decode(final_tokens)