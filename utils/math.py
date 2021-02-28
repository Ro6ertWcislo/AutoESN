from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def spectral_normalize(tensor: Tensor) -> Tensor:
    u, s, v = torch.svd(tensor, compute_uv=False)
    norm = torch.max(s)
    return tensor / norm


def linear(input: Tensor, hx: Tensor, weight_ih: Tensor, weight_hh: Tensor, bias_ih: Optional[Tensor],
           bias_hh: Optional[Tensor]) -> Tensor:
    return F.linear(input, weight_ih, bias_ih) + F.linear(hx, weight_hh, bias_hh)


def leaky(hx_prev: Tensor, hx_next: Tensor, leaky_rate: float = 1.) -> Tensor:
    return (1. - leaky_rate) * hx_prev + leaky_rate * hx_next

