import torch

from fmengine.core.configs.train_config import OptimizerConfig


def build_optimizer(model_parts, optimizer_config: OptimizerConfig):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model):
        name = optimizer_config.name
        lr = optimizer_config.lr
        fused = optimizer_config.fused

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": lr,
            "betas": optimizer_config.betas,
            "weight_decay": optimizer_config.weight_decay,
            "fused": optimizer_config.fused,
            "foreach": not fused,
        }
        if name == "adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

        return optimizer

    class OptimizersContainer:
        """Util for calling step/zero_grad on multiple optimizers needed for virtual pipeline stages"""

        def __init__(self, optimizers):
            self.optimizers = optimizers

        def step(self):
            for optimizer in self.optimizers:
                optimizer.step()

        def zero_grad(self):
            for optimizer in self.optimizers:
                optimizer.zero_grad()

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])
