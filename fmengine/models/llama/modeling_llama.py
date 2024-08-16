import torch
import torch.nn as nn
from torch.distributed import DeviceMesh

from fmengine.core.nn import (
    FeedForward,
    Llama3ScaledRoPE,
    CausalSelfAttention,
    TransformerDecoder,
    TransformerDecoderLayer,
    TiedEmbeddingTransformerDecoder,
    RMSNorm,
)
from fmengine.core.parallelism.parallel_dims import ParallelDims
from fmengine.core.parallelism.parallelizer import apply_tp
from fmengine.core.configs.train_config import TrainingConfig

from .config_llama import LlamaArgs

def build_llama_3(args: LlamaArgs):
    head_dim = args.hidden_size // args.n_heads
    num_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
    rope = Llama3ScaledRoPE(
        dim=args.hidden_size,
        max_seq_len=args.max_seq_len,
        base=args.rope_theta
    )
    self_attn = CausalSelfAttention(
        args.n_heads,
        num_kv_heads,
        head_dim,
        q_proj=nn.Linear(args.hidden_size, args.n_heads *
                         head_dim, bias=False),
        k_proj=nn.Linear(args.hidden_size, args.n_kv_heads *
                         head_dim, bias=False),
        v_proj=nn.Linear(args.hidden_size, args.n_kv_heads *
                         head_dim, bias=False),
        output_proj=nn.Linear(args.hidden_size, args.hidden_size, bias=False),
        pos_embeddings=rope,
        max_seq_len=args.max_seq_len,
        attn_dropout=args.attn_dropout,
    )
    hidden_dim = args.intermediate_dim if args.intermediate_dim is not None else args.hidden_size * \
        args.ffn_dim_multiplier

    # build mlp module...
    mlp = FeedForward(dim=args.hidden_size, hidden_dim=hidden_dim, activation=args.activation)
    
    layer = TransformerDecoderLayer(
        attn=self_attn,
        mlp=mlp,
        self_attn_norm=RMSNorm(args.hidden_size, args.norm_eps),
        mlp_norm=RMSNorm(args.hidden_size, args.norm_eps),
    )
    tok_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
    output_proj = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
    return TransformerDecoder(
        tok_embeddings=tok_embeddings,
        layer=layer,
        num_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        num_heads=args.n_heads,
        head_dim=head_dim,
        norm=RMSNorm(args.hidden_size, args.norm_eps),
        output=output_proj,
    )

def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    train_config: TrainingConfig
):
    if parallel_dims.tp_enabled:
        if (
            train_config.enable_async_tensor_parallel
            and not train_config.training.compile
        ):
            raise RuntimeError("Async TP requires --training.compile")
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_float8=train_config.float8.enable_float8_linear,
            enable_async_tp=train_config.enable_async_tensor_parallel,
        )