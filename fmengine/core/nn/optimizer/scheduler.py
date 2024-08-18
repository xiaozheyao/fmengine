import functools

from torch.optim.lr_scheduler import LambdaLR

from fmengine.core.configs.train_config import TrainJobConfig


def linear_warmup_linear_decay(warmup_steps: int, decay_steps: int, current_step: int) -> float:
    """Computes linear warmup followed by linear decay.
    Per LambdaLR requirement, this is accomplished by returning
    a multiplicative factor to adjust the learning rate to
    create the desired schedule.
    """
    if current_step < warmup_steps:
        # linear warmup
        # 0-indexed step, hence + 1 adjustments
        current_step += 1
        curr_adjustment = float(current_step / (warmup_steps + 1))

    else:
        # linear decay
        normalized_step = decay_steps - (current_step - warmup_steps)
        curr_adjustment = 1 - (decay_steps - normalized_step) / decay_steps

    return curr_adjustment


def build_lr_scheduler(optimizers, train_job_config: TrainJobConfig):
    def _build_lr_scheduler(optimizer):
        """Build a linear warmup and linear decay scheduler"""
        warmup_steps = int(train_job_config.training.warmup_steps)
        decay_steps = float(max(1, train_job_config.training.train_steps - warmup_steps))
        lr_lambda = functools.partial(linear_warmup_linear_decay, warmup_steps, decay_steps)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return warmup_scheduler

    class SchedulersContainer:
        """Util for calling step on multiple learning rate schedulers needed for virtual pipeline stages"""

        def __init__(self, schedulers):
            self.schedulers = schedulers

        def step(self):
            for schedulers in self.schedulers:
                schedulers.step()

    return SchedulersContainer([_build_lr_scheduler(optimizer) for optimizer in optimizers])
