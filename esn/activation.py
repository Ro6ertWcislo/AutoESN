import torch
from torch.nn import functional as F


# todo implement as classes

def linear(input, hx, weight_ih, weight_hh, bias_ih, bias_hh): # todo make it bias aware
    return F.linear(input, weight_ih, bias_ih) + F.linear(hx, weight_hh, bias_hh)


def leaky(leaky_rate, hx_prev, hx_next):
    return (1 - leaky_rate) * hx_prev + leaky_rate * hx_next


def tanh(input, hx, weight_ih, weight_hh, bias_ih, bias_hh):
    return 0.7*hx + 0.3* torch.tanh(
        linear(input, hx, weight_ih, weight_hh, bias_ih, bias_hh)
    )

def SelfReg(input, hx, weight_ih, weight_hh, bias_ih, bias_hh, Sr=0.9):
    return Sr * spectral_norm(
        linear(input, hx, weight_ih, weight_hh, bias_ih, bias_hh)
    )

# todo add leaky implementations

def spectral_norm(tensor):
    u, s, v = torch.svd(tensor, compute_uv=False)
    norm = s[0]  # max eigenvalue is on 1 place
    return tensor / norm