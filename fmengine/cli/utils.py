import os
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
