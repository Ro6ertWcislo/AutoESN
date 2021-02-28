import torch
from torch import nn, Tensor

from utils import math as M
from utils.types import ActivationFunction


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


def linear(leaky_rate: float = 1.0) -> Activation:
    def id(input: Tensor) -> Tensor:
        return input

    return Activation(activation_function=id, leaky_rate=leaky_rate)


def self_normalizing(leaky_rate: float = 1.0, spectral_radius: float = 0.9) -> Activation:
    def activation_function(input: Tensor) -> Tensor:
        return spectral_radius * M.spectral_normalize(input)

    return Activation(activation_function=activation_function, leaky_rate=leaky_rate)


def self_normalizing_default(leaky_rate: float = 0.9, spectral_radius: float = 100.0) -> Activation:
    def activation_function(input: Tensor) -> Tensor:
        return spectral_radius * (input / torch.norm(input))

    return Activation(activation_function=activation_function, leaky_rate=leaky_rate)
