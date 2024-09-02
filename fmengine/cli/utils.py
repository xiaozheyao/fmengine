import os
import torch
import typer
from omegaconf import OmegaConf
from dacite import from_dict
from fmengine.core.configs.train_config import TrainJobConfig


def parse_train_config(config: str):
    if not os.path.exists(config):
        raise ValueError(f"Config file not found: {config}")
    # if it is a directory, search for yaml files
    if os.path.isdir(config):
        config_files = [os.path.join(config, f) for f in os.listdir(config) if f.endswith(".yaml")]
        typer.echo(f"config files found: {config_files}")
        configs = [OmegaConf.load(f) for f in config_files]
        config = OmegaConf.merge(*configs)
    elif os.path.isfile(config):
        config = OmegaConf.load(config)
    config = OmegaConf.to_container(config)
    config = from_dict(data_class=TrainJobConfig, data=config)
    return config


def enforce_nondistributed_env():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == 1, "Exporting is only supported in single GPU mode"
    # ensures calling this function with python main.py instead of torch.distributed.launch (i.e. torchrun)
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9090"


def multinomial_sample_one(probs: torch.Tensor) -> torch.Tensor:
    """Samples from a multinomial distribution."""
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(logits: torch.Tensor, temperature=0.9, top_k: int = 50) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # select the very last value from the top_k above as the pivot
        pivot = v.select(-1, -1).unsqueeze(-1)
        # set everything smaller than pivot value to inf since these
        # should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return multinomial_sample_one(probs)
