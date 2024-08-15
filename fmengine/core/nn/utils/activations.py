from typing import Callable

import torch.nn.functional as F

ACTIVATIONS = {}


def register_activation(act_fn: Callable):
    ACTIVATIONS[act_fn.__name__] = act_fn


@register_activation
def silu(x):
    return F.silu(x)


def get_activation_fn(activation: str):
    return ACTIVATIONS[activation] if activation in ACTIVATIONS else getattr(F, activation)
