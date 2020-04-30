import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Optional


def spectral_norm(tensor):
    u, s, v = torch.svd(tensor, compute_uv=False)
    norm = s[0]  # max eigenvalue is on 1 place
    return tensor / norm


def eigen_norm(tensor):
    abs_eigs = (torch.eig(tensor)[0] ** 2).sum(1).sqrt()
    return tensor / torch.max(abs_eigs)


def linear(input: Tensor, hx: Tensor, weight_ih: Tensor, weight_hh: Tensor, bias_ih: Optional[Tensor],
           bias_hh: Optional[Tensor]) -> Tensor:
    return F.linear(input, weight_ih, bias_ih) + F.linear(hx, weight_hh, bias_hh)


def leaky(hx_prev, hx_next, leaky_rate=1.0):
    return (1 - leaky_rate) * hx_prev + leaky_rate * hx_next
