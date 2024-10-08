import torch
from fmengine.core.configs.train_config import AutoOptimizationFlags

optimizations_flags = AutoOptimizationFlags()


def _patch_with_transformer_engine():
    try:
        import transformer_engine as te
    except:
        print("Transformer Engine not found, skipping patching.")
        return False
    torch.nn.Linear = te.pytorch.Linear
    return True


def auto_patch(enable_te: False) -> AutoOptimizationFlags:
    if enable_te:
        optimizations_flags.use_transformer_engine = _patch_with_transformer_engine()
    return optimizations_flags
