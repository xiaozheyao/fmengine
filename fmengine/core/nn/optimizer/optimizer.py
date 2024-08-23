import torch

from fmengine.core.configs.train_config import OptimizerConfig


def build_optimizer(model_parts, optimizer_config: OptimizerConfig):
    """Wrap one optimizer per model part in an OptimizersContainer which provides a single
    step() and zero_grad() method for all the child optimizers.
    """

    def _build_optimizer(model):
        name = optimizer_config.name

        # Common parameters for both optimizers
        optimizer_kwargs = {
            "lr": optimizer_config.lr,
            "betas": optimizer_config.betas,
            "weight_decay": optimizer_config.weight_decay,
            "fused": optimizer_config.fused,
            "foreach": not optimizer_config.fused,
        }
        if name == "adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        elif name == "apex_adam":
            raise NotImplementedError("Apex Adam optimizer is not supported yet.")
            from apex.optimizers import FusedAdam
            optimizer_kwargs.pop("fused")
            optimizer_kwargs.pop("foreach")
            # apex optimizers require models to be on gpu
            if not torch.cuda.is_available():
                raise ValueError("Apex optimizers require models to be on GPU, but torch.cuda is not available.")
            model = model.to("cuda")
            optimizer = FusedAdam(model.parameters(), **optimizer_kwargs)
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

        @property
        def learning_rate(self):
            return self.optimizers[0].param_groups[0]["lr"]

    return OptimizersContainer([_build_optimizer(model) for model in model_parts])
