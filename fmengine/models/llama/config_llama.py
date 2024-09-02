from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaArgs:
    architecture: str = "llama"
    hidden_size: int = 4096
    n_layers: int = 22
    n_heads: int = 32
    intermediate_dim: Optional[int] = 11008
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    norm_type: str = "fused_rmsnorm"
    activation: str = "silu"
    attn_dropout: float = 0.0
    # torch dtype only specifies the model weights dtype
    # actual computation is done in bfloat16 with torch.autocast
    torch_dtype: str = "float32"
    # now below are fixed and the values here are not used
    initializer_range: float = 0.02
