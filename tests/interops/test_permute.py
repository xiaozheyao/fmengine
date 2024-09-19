import torch
from unittest import TestCase
from fmengine.models.llama.interop_llama import permute, inverse_permute


class TestPermute(TestCase):
    def test_permute(self):
        dim1 = 8
        dim2 = 4
        n_heads = 2
        w = torch.randn(dim1, dim2)
        w_permuted = permute(w, n_heads, dim1, dim2)
        w_inversed = inverse_permute(w_permuted, n_heads, dim1, dim2)
        self.assertTrue(torch.allclose(w, w_inversed))
