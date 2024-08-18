from typing import List

from transformers import AutoTokenizer

from .tokenizer import Tokenizer


class HFTokenizer(Tokenizer):
    def __init__(self, tokenizer_name_or_path: str):
        super().__init__(tokenizer_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    def encode(self, s: str, **kwargs) -> List[int]:
        assert type(s) == str, f"Input must be a string, got {type(s)}"
        return self.tokenizer.encode(s, **kwargs)

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)
