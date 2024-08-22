import typer
from fmengine.cli.trainer import train_entry
from fmengine.cli.utils import parse_train_config
from fmengine.cli.export import export_entry
from fmengine.cli.eval import evaluation_entry

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
def eval(
    model_id: str = typer.Option(..., help="Path to the model file"),
    tasks: str = typer.Option(..., help="Comma-separated tasks to evaluate"),
):
    evaluation_entry(model_id, tasks=tasks)


@fmengine.command()
def inference(
    model_id: str = typer.Option(..., help="Path to the model file"),
    prompt: str = typer.Option(..., help="Prompt to generate text"),
    temperature: float = typer.Option(0.5, help="Temperature for sampling"),
    top_k: int = typer.Option(50, help="Top k for sampling"),
    top_p: float = typer.Option(0.9, help="Top p for sampling"),
):
    import torch
    import transformers

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        do_sample=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
        max_new_tokens=128,
    )
    output = pipeline(
        prompt,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    print(output)


if __name__ == "__main__":
    fmengine()
