import os
from groq import Groq
from src.strings import (
    DRAFT_MODEL_NAME,
    ENV_GROQ_API_KEY,
    ERROR_GROQ_API_KEY,
    DRAFT_MAX_TOKENS,
    DRAFT_TEMPERATURE,
    USER_ROLE
)

class DraftModel:
    def __init__(self, model_name=DRAFT_MODEL_NAME):
        api_key = os.getenv(ENV_GROQ_API_KEY)
        if not api_key:
            raise ValueError(ERROR_GROQ_API_KEY)
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt, max_new_tokens=DRAFT_MAX_TOKENS):
        """
        Generate draft response using faster, smaller model via API.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": USER_ROLE, "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=DRAFT_TEMPERATURE
        )
        return response.choices[0].message.content
