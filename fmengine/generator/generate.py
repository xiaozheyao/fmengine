import torch
from torch import nn
from fmengine.data.tokenizer import Tokenizer
from .sampler import sample


def generate(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.6,
    top_k: int = 50,
):
    model.eval()
    prompt = tokenizer.encode(prompt, return_tensors="pt").cuda()
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    bsz, prompt_length = prompt.size()
    generated_tokens = prompt.clone()

    input_pos = torch.arange(0, model.max_seq_len, device=prompt.device)
    for _ in range(max_tokens):
        logits = model(generated_tokens, input_pos=input_pos[:prompt_length])[:, -1]
        next_token = sample(logits, temperature, None)
        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
        prompt_length += 1
    output = tokenizer.decode(generated_tokens[0].tolist())
    return output
