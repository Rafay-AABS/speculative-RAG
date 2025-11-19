import os
from groq import Groq

class DraftModel:
    def __init__(self, model_name="llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found. Get free API key from https://console.groq.com")
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt, max_new_tokens=512):
        """
        Generate draft response using faster, smaller model via API.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
