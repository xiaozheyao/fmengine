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
from fmengine.generator import sample


def generate_next_token(
    model: nn.Module,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = None,
) -> torch.Tensor:
    """Generates the next tokens."""
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos)[:, -1]
    return logits, sample(logits, 1e-5, None)


@torch.inference_mode()
def generate_entry(
    job_config: TrainJobConfig, prompt: str, max_tokens: int = 128, top_k: int = 50, temperature: float = 0.5
):
    ao_flags = auto_patch()
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

    prompt = tokenizer.encode(prompt, return_tensors="pt").cuda()
    print(f"Prompt: {prompt}")
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()

    input_pos = torch.arange(0, model.max_seq_len, device=prompt.device)
    logits, tokens = generate_next_token(
        model,
        input_pos=input_pos[:prompt_length],
        x=prompt,
        temperature=temperature,
        top_k=top_k,
    )
    print(f"Logits after first iteration: {logits.shape}")
    print(f"Logits after first iteration: {logits}")

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
    curr_pos = prompt_length
    for _ in range(max_tokens - 1):
        curr_input_pos = input_pos[: curr_pos + 1]
        logits, tokens = generate_next_token(
            model,
            input_pos=curr_input_pos,
            x=generated_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1
    decoded_output = tokenizer.decode(generated_tokens[0])
    torch.distributed.destroy_process_group()
    return decoded_output
