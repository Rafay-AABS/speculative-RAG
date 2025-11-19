from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class DraftModel:
    def __init__(self, model_name="Qwen/Qwen2.5-0.5B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def generate(self, prompt, max_new_tokens=80):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        output = self.model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=True, top_k=20
        )
        tokens = output[0][input_ids.shape[1]:]  # newly generated
        return tokens
