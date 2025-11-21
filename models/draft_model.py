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
from src.exceptions import ModelError
from src.logger import logger


class DraftModel:
    def __init__(self, model_name: str = DRAFT_MODEL_NAME):
        api_key = os.getenv(ENV_GROQ_API_KEY)
        if not api_key:
            raise ModelError(ERROR_GROQ_API_KEY)
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized draft model: {model_name}")

    def generate(self, prompt: str, max_new_tokens: int = DRAFT_MAX_TOKENS) -> str:
        """
        Generate draft response using faster, smaller model via API.
        """
        try:
            logger.debug(f"Generating draft with {self.model_name}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": USER_ROLE, "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=DRAFT_TEMPERATURE
            )
            text = response.choices[0].message.content
            logger.debug(f"Draft generated: {len(text)} characters")
            return text
        except Exception as e:
            logger.error(f"Draft model generation failed: {e}")
            raise ModelError(f"Draft model generation failed: {e}") from e
