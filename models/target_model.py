import os
from groq import Groq

class TargetModel:
    def __init__(self, model_name="llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Get free API key from https://console.groq.com")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def verify(self, prompt, draft_text):
        """
        Verify draft output by generating with target model and comparing.
        In API-based speculative decoding, we generate and check if draft matches.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        verified_text = response.choices[0].message.content
        
        # Simple verification: if draft is prefix of target output, accept it
        if verified_text.startswith(draft_text.strip()):
            return draft_text
        else:
            return verified_text
