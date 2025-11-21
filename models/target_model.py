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
from src.exceptions import ModelError
from src.logger import logger


class TargetModel:
    def __init__(self, model_name: str = TARGET_MODEL_NAME):
        api_key = os.getenv(ENV_GROQ_API_KEY)
        if not api_key:
            raise ModelError(ERROR_GROQ_API_KEY)
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized target model: {model_name}")

    def verify(self, prompt: str, draft_text: str) -> str:
        """
        Verify draft output by generating with target model and comparing.
        In API-based speculative decoding, we generate and check if draft matches.
        """
        try:
            logger.debug(f"Verifying with {self.model_name}")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": USER_ROLE, "content": prompt}],
                max_tokens=TARGET_MAX_TOKENS,
                temperature=TARGET_TEMPERATURE
            )
            verified_text = response.choices[0].message.content
            
            # Simple verification: if draft is prefix of target output, accept it
            if verified_text.startswith(draft_text.strip()):
                logger.info("Draft accepted by target model")
                return draft_text
            else:
                logger.info("Target model provided different output")
                return verified_text
        except Exception as e:
            logger.error(f"Target model verification failed: {e}")
            raise ModelError(f"Target model verification failed: {e}") from e
        else:
            return verified_text
