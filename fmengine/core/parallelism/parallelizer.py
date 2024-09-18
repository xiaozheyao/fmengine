from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard, CPUOffloadPolicy
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper as ptd_checkpoint_wrapper
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from fmengine.utilities.logging import logger


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8: bool = False,
    enable_async_tp: bool = False,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears
    if enable_float8:
        # TODO(vkuzo): once float8 configuration supports delayed scaling,
        # add a check here to enforce supported float8 all-gather configurations
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for i, transformer_block in enumerate(model.layers):
        layer_plan = {
            "self_attn_norm": SequenceParallel(),
            "attn": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attn.q_proj": colwise_parallel(),
            "attn.k_proj": colwise_parallel(),
            "attn.v_proj": colwise_parallel(),
            "attn.output_proj": rowwise_parallel(output_layouts=Shard(1)),
            "mlp_norm": SequenceParallel(),
            "mlp": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "mlp.w1": colwise_parallel(),
            "mlp.w2": rowwise_parallel(output_layouts=Shard(1)),
            "mlp.w3": colwise_parallel(),
        }

        model.layers[i] = parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

        torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)

    logger.info(
        f"Applied {'Float8 ' if enable_float8 else ''}{'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
}


def _apply_ac_to_transformer_block(module: nn.Module, ac_mode: str, selective_ac_option: str):
    valid_ac_modes = ("full", "selective")
    if ac_mode not in valid_ac_modes:
        raise ValueError(f"Invalid AC mode: {ac_mode}. Valid modes: {valid_ac_modes}")

    if ac_mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_mode == "selective", f"{ac_mode}"
    use_op_sac = selective_ac_option == "op"
    use_layer_sac = selective_ac_option.isdigit()
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)
                return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:
        # Checkpoint every `ac_freq` of the modules passed to this function
        ac_freq = int(selective_ac_option)
        ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
        ptd_checkpoint_wrapper._count += 1
        if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
            return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
        else:
            return module


def apply_ac(model: nn.Module, ac_mode: str, selective_ac_option: str = "2"):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(transformer_block, ac_mode, selective_ac_option)
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_mode} activation checkpointing to the model")


def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    logger.info(f"torch.compile() enabled")
    # for layer_id, transformer_block in model.layers.named_children():
    #     transformer_block = torch.compile(transformer_block, fullgraph=True)
    #     model.layers.register_module(layer_id, transformer_block)

    # logger.info("Compiling each TransformerBlock with torch.compile")

    import os

    backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
    model.compile(backend=backend, fullgraph=True)


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    tp_enabled: bool,
    pp_enabled: bool,
    cpu_offload: bool = False,
):
    """
    Apply data parallelism to the model. FSDP2 is used here.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    offload_policy = CPUOffloadPolicy(pin_memory=True)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # TODO: remove this check once PyTorch 2.5 is released. We can safely assume
    # that users won't use a nightly build which is older than 20240809 by then.
    if tp_enabled:
        # check if strided sharding is enabled, which is necessary for 2D/3D DCP
        raise ValueError(f"TP + FSDP is not supported yet")

    for layer_id, transformer_block in enumerate(model.layers):
        if pp_enabled:
            # For PP, do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = False
        else:
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.layers) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
            offload_policy=offload_policy if cpu_offload else None,
        )
    fully_shard(
        model,
        **fsdp_config,
        reshard_after_forward=not pp_enabled,
        offload_policy=offload_policy if cpu_offload else None,
    )
    logger.info("Applied FSDP to the model")


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = "python_reducer_without_compiled_forward"
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
