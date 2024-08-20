import typer
from fmengine.cli.trainer import train_entry
from fmengine.cli.utils import parse_train_config
from fmengine.cli.export import export_entry

fmengine = typer.Typer()

@fmengine.command()
def train(config: str = typer.Option(..., help="Path to the config file")):
    # check if the config file exists
    typer.echo(f"Training with config: {config}")
    config = parse_train_config(config)
    train_entry(config)

@fmengine.command()
def export(
    ckpt_path: str = typer.Option(..., help="Path to the checkpoint file"),
    step: int = typer.Option(-1, help="Step to export the model"),
    config: str = typer.Option(..., help="Path to the config file"),
    output_path: str = typer.Option(..., help="Path to the output directory"),
):
    config = parse_train_config(config)
    export_entry(ckpt_path, step, config, output_path)

@fmengine.command()
def inference(
    model_id: str = typer.Option(..., help="Path to the model file"),
    prompt: str = typer.Option(..., help="Prompt to generate text"),
):
    import torch
    import transformers
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        max_new_tokens=128,
    )
    output = pipeline(prompt)
    print(output)
    
if __name__ == "__main__":
    fmengine()
