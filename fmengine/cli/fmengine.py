import os

import typer
from omegaconf import OmegaConf

fmengine = typer.Typer()

@fmengine.command()
def train(config: str = typer.Option(..., help="Path to the config file")):

    # check if the config file exists
    typer.echo(f"Training with config: {config}")
    if not os.path.exists(config):
        raise ValueError(f"Config file not found: {config}")
    # if it is a directory, search for yaml files
    if os.path.isdir(config):
        config_files = [
            os.path.join(config, f) for f in os.listdir(config) if f.endswith(".yaml")
        ]
        typer.echo(f"Multiple config files found: {config_files}")
        configs = [OmegaConf.load(f) for f in config_files]
        config = OmegaConf.merge(*configs)
    elif os.path.isfile(config):
        config = OmegaConf.load(config)
    typer.echo(f"Config: {config}")


@fmengine.command()
def inspect():
    pass


if __name__ == "__main__":
    fmengine()
