import os
from abc import ABC, abstractmethod
from typing import List


class Tokenizer(ABC):
    # basic tokenizer interface, for typing purpose mainly
    def __init__(self, tokenizer_path: str):
        assert os.path.exists(tokenizer_path), f"The tokenizer path does not exist: {tokenizer_path}"
        self._n_words = -1

    @abstractmethod
    def encode(self, *args, **kwargs) -> List[int]: ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str: ...

    @property
    def n_words(self) -> int:
        return self._n_words
