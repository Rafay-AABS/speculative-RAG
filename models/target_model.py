import os
from groq import Groq
from src.strings import (
    TARGET_MODEL_NAME,
    ENV_GROQ_API_KEY,
    ERROR_GROQ_API_KEY,
    TARGET_MAX_TOKENS,
    TARGET_TEMPERATURE,
    USER_ROLE
)

class TargetModel:
    def __init__(self, model_name=TARGET_MODEL_NAME):
        api_key = os.getenv(ENV_GROQ_API_KEY)
        if not api_key:
            raise ValueError(ERROR_GROQ_API_KEY)
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def verify(self, prompt, draft_text):
        """
        Verify draft output by generating with target model and comparing.
        In API-based speculative decoding, we generate and check if draft matches.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": USER_ROLE, "content": prompt}],
            max_tokens=TARGET_MAX_TOKENS,
            temperature=TARGET_TEMPERATURE
        )
        verified_text = response.choices[0].message.content
        
        # Simple verification: if draft is prefix of target output, accept it
        if verified_text.startswith(draft_text.strip()):
            return draft_text
        else:
            return verified_text
