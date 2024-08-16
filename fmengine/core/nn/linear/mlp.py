from torch import Tensor, nn

from fmengine.core.nn.utils.activations import get_activation_fn


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        with_bias: bool = False,
        activation: str = "silu",
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=with_bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=with_bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=with_bias)
        self.activation = get_activation_fn(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.activation(self.w1(x)) * self.w3(x))
