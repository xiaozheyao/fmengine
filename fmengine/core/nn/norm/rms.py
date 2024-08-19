import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6, compile: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.rmsnorm_fn = torch.compile(self.compute_rmsnorm, fullgraph=True) if compile else self.compute_rmsnorm

    @staticmethod
    def compute_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        def _norm(x, eps):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

        output = _norm(x.float(), eps).type_as(x)
        return output * weight

    def forward(self, x: torch.Tensor):
        return self.rmsnorm_fn(x, self.weight, self.eps)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore
