import os
import re
import torch
import torch.distributed.checkpoint as dcp

from fmengine.core.configs.train_config import TrainJobConfig
from fmengine.models.builder import build_model, export_to_huggingface
from fmengine.utilities import logger
from fmengine.core.parallelism.distributed import init_distributed
from fmengine.datasets.tokenizer import build_tokenizer


def export_entry(ckpt_path: str, step: int, job_config: TrainJobConfig, output_path: str):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert world_size == 1, "Exporting is only supported in single GPU mode"
    # ensures calling this function with python main.py instead of torch.distributed.launch
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "9090"

    init_distributed(dump_folder=job_config.training.dump_folder)
    with torch.device("meta"):
        model = build_model(job_config.model)
    model.to_empty(device="cpu")
    if step == -1:
        step_counts = []
        for filename in os.listdir(ckpt_path):
            match = re.search(r"step-(\d+)", filename)
            metadata_probe = os.path.join(ckpt_path, filename, ".metadata")
            if match and os.path.isfile(metadata_probe):
                step_counts.append(int(match.group(1)))
        if not step_counts:
            raise ValueError(f"No valid checkpoint found in {ckpt_path}")
        step = max(step_counts)
    states = {"model": model.state_dict()}
    logger.info(f"Loading the checkpoint at step {step}")
    tokenizer = build_tokenizer(job_config.tokenizer.tokenizer_type, job_config.tokenizer.tokenizer_name_or_path)
    dcp.load(states, checkpoint_id=os.path.join(ckpt_path, f"step-{step}"))
    
    model, hf_config = export_to_huggingface(states, job_config.model)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model.save_pretrained(output_path)
    hf_config.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    torch.distributed.destroy_process_group()
    return True
