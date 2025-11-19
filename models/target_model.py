from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class TargetModel:
    def __init__(self, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )

    def verify(self, prompt, draft_tokens):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        tgt_len = len(draft_tokens)

        target_output = self.model(input_ids).logits[:, -1, :]
        accepted = []

        for i, token in enumerate(draft_tokens):
            target_probs = torch.argmax(target_output)
            if token == target_probs:
                accepted.append(token)
            else:
                regen = self.model.generate(
                    input_ids, max_new_tokens=tgt_len
                )[0][input_ids.shape[1]:]
                return accepted + regen

        return accepted
