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
