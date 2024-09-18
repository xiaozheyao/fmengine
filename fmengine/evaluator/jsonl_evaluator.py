import json
from pathlib import Path
from torch import nn


class JSONLEvaluator:

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        # check if file exists
        if not self.filepath.exists():
            raise FileNotFoundError(f"File {self.filepath} not found")
        with open(self.filepath, "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def evaluate(self, model: nn.Module, tokenizer):
        pass
