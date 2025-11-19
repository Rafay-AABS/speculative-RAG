from src.rag_prompt import build_rag_prompt
from src.speculative_decoder import speculative_decode
from models.draft_model import DraftModel
from models.target_model import TargetModel

class SpeculativeRAG:
    def __init__(self, retriever):
        self.retriever = retriever
        self.draft = DraftModel()
        self.target = TargetModel()

    def run(self, query, texts):
        docs = self.retriever.retrieve(query, texts)
        prompt = build_rag_prompt(query, docs)

        answer = speculative_decode(prompt, self.draft, self.target)
        return answer
