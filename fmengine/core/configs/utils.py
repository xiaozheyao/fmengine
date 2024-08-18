import torch
from omegaconf import OmegaConf


def dict_to_config(d: dict):
    """
    Convert a dictionary to a config object.
    """
    return OmegaConf.structured()


TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
