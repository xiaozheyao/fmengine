import torch
from typing import Any, List, Mapping, Protocol


class Transform(Protocol):
    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        pass
