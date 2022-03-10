import torch
from torch import nn, Tensor

from auto_esn.utils import math as M
from auto_esn.utils.types import ActivationFunction


class Activation(nn.Module):
    def __init__(self, activation_function: ActivationFunction, leaky_rate: float):
        super().__init__()
        self.leaky_rate = leaky_rate
        self.activation_function = activation_function

    def forward(self, pre_activation: Tensor, prev_state: Tensor = None) -> Tensor:
        hx_next = self.activation_function(pre_activation)
        # leaky rate == 1.0 means no leaky_rate at all. hx_prev gets zeroed
        return M.leaky(hx_prev=prev_state, hx_next=hx_next, leaky_rate=self.leaky_rate)


def tanh(leaky_rate: float = 1.0) -> Activation:
    return Activation(activation_function=torch.tanh, leaky_rate=leaky_rate)


def relu(leaky_rate: float = 1.0) -> Activation:
    return Activation(activation_function=torch.relu, leaky_rate=leaky_rate)


class Identity:
    def __call__(self, input: Tensor):
        return input


def linear(leaky_rate: float = 1.0) -> Activation:
    return Activation(activation_function=Identity(), leaky_rate=leaky_rate)


class SelfNorm:
    def __init__(self, spectral_radius: float = 100.0):
        self.spectral_radius = spectral_radius

    def __call__(self, input: Tensor):
        return self.spectral_radius * (input / torch.norm(input))


def self_normalizing_default(leaky_rate: float = 0.9, spectral_radius: float = 100.0) -> Activation:
    return Activation(activation_function=SelfNorm(spectral_radius), leaky_rate=leaky_rate)
