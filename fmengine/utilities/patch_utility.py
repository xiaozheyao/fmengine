import torch


def _patch_with_transformer_engine():
    try:
        import transformer_engine as te
    except ImportError:
        print("Transformer Engine not found, skipping patching.")
        return
    torch.nn.Linear = te.pytorch.Linear


def auto_patch():
    _patch_with_transformer_engine()
