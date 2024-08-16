from dataclasses import dataclass
from typing import Optional, Tuple

from fmengine.core.nn.utils import (apply_rotary_emb, precompute_freqs_cis,
                                    repeat_kv, reshape_for_broadcast)


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
    max_batch_size: int = 32
    max_seq_len: int = 2048
    # If `True`, then each transformer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True
    norm_type: str = "rmsnorm"
    activation: str = "silu"
    attn_dropout: float = 0.0
