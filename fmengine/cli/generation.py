import re
import os
import torch
import torch.distributed.checkpoint as dcp
from torch import nn
from fmengine.models.builder import build_model
from fmengine.core.configs.train_config import TrainJobConfig
from fmengine.utilities import auto_patch
from fmengine.cli.utils import enforce_nondistributed_env
from fmengine.core.parallelism.distributed import init_distributed
from fmengine.data.tokenizer import build_tokenizer
from fmengine.generator import generate


@torch.inference_mode()
def generate_entry(
    job_config: TrainJobConfig, prompt: str, max_tokens: int = 128, top_k: int = 50, temperature: float = 0.5
):
    ao_flags = auto_patch(job_config.auto_patch.use_transformer_engine)
    enforce_nondistributed_env()
    init_distributed(dump_folder=job_config.training.dump_folder)
    with torch.device("meta"):
        model = build_model(job_config.model, ao_flags)
    model.to_empty(device="cpu")
    step_counts = []
    ckpt_path = job_config.checkpoint.ckpt_dir
    for filename in os.listdir(ckpt_path):
        match = re.search(r"step-(\d+)", filename)
        metadata_probe = os.path.join(ckpt_path, filename, ".metadata")
        if match and os.path.isfile(metadata_probe):
            step_counts.append(int(match.group(1)))
    step = max(step_counts)
    if not step_counts:
        raise ValueError(f"No valid checkpoint found in {ckpt_path}")

    states = {"model": model.state_dict()}
    print(f"Loading the checkpoint at step {step}")
    tokenizer = build_tokenizer(job_config.tokenizer.tokenizer_type, job_config.tokenizer.tokenizer_name_or_path)
    dcp.load(states, checkpoint_id=os.path.join(ckpt_path, f"step-{step}"))
    model.load_state_dict(states["model"], strict=True)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    model.eval()

    decoded_output = generate(model, tokenizer, prompt, max_tokens, temperature, top_k)
    return decoded_output
