import json
from pathlib import Path
from torch import nn
from fmengine.generator import generate
from fmengine.data.tokenizer import Tokenizer


class JSONLEvaluator:

    def __init__(self, filepath: str) -> None:
        self.filepath = Path(filepath)
        # check if file exists
        if not self.filepath.exists():
            raise FileNotFoundError(f"File {self.filepath} not found")
        with open(self.filepath, "r") as f:
            self.data = [json.loads(line) for line in f.readlines()]

    def __len__(self):
        return len(self.data)

    def evaluate(self, model: nn.Module, tokenizer: Tokenizer):
        results = []
        for datum in self.data:
            output = generate(model, tokenizer, datum["prompt"])
            results.append({"prompt": datum["prompt"], "output": output})
        return results
