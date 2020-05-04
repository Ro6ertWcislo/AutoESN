from typing import List, Optional

import torch.nn
from torch import Tensor

from utils.math import spectral_normalize
from utils.types import Initializer


def _scale(factor: float = 1.0) -> Initializer:
    def __scale(weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        return weight * factor

    return __scale


def _normalize() -> Initializer:
    def __normalize(weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        return spectral_normalize(weight)

    return __normalize


def _sparse(density: float = 1.0) -> Initializer:
    def __sparse(weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        zero_mask = torch.empty_like(weight).uniform_() > (1 - density)
        return weight * zero_mask

    return __sparse


def _uniform(min_val: float = -1, max_val: float = 1) -> Initializer:
    def __uniform(weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        new_weight = torch.empty_like(weight)
        torch.nn.init.uniform_(new_weight, min_val, max_val)
        return new_weight

    return __uniform


def _xavier_uniform() -> Initializer:
    def __uniform(weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        new_weight = torch.empty_like(weight)
        torch.nn.init.xavier_uniform_(new_weight)
        return new_weight

    return __uniform


def _spectral_noisy(spectral_radius=0.9, noise_magnitude=0.2) -> Initializer:
    def __spectral_noisy(weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        def scale(tensor):
            return (tensor / tensor.size(0)) * spectral_radius

        w_hh = scale(torch.ones_like(weight))
        noise = scale(torch.rand_like(w_hh)) * noise_magnitude
        return w_hh + (noise - torch.mean(noise))

    return __spectral_noisy


class CompositeInitializer(object):
    def __init__(self):
        self.initializers: List[Initializer, ...] = []

    def scale(self, factor: float = 1.0):
        self.initializers.append(_scale(factor=factor))
        return self

    def normalize(self):
        self.initializers.append(_normalize())
        return self

    def sparse(self, density=1.0):
        self.initializers.append(_sparse(density=density))
        return self

    def uniform(self, min_val: float = -1, max_val: float = 1):
        self.initializers.append(_uniform(min_val=min_val, max_val=max_val))
        return self

    def xavier_uniform(self):
        self.initializers.append(_xavier_uniform())
        return self

    def spectral_noisy(self, spectral_radius=0.9, noise_magnitude=0.2):
        self.initializers.append(_spectral_noisy(spectral_radius=spectral_radius, noise_magnitude=noise_magnitude))
        return self

    def __call__(self, weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        for initializer in self.initializers:
            weight = initializer(weight, reference_weight)
        return weight


def uniform(min_val=-1, max_val=1) -> Initializer:
    return CompositeInitializer().uniform(min_val=min_val, max_val=max_val)


def default_hidden(density=0.1, spectral_radius=0.9):
    return CompositeInitializer() \
        .xavier_uniform() \
        .sparse(density=density) \
        .normalize() \
        .scale(factor=spectral_radius)


class WeightInitializer(object):
    def __init__(self,
                 weight_ih_init: Initializer = uniform(),
                 weight_hh_init: Initializer = default_hidden(),
                 bias_ih_init: Initializer = uniform(),
                 bias_hh_init: Initializer = uniform()):
        self.weight_ih_init = weight_ih_init
        self.weight_hh_init = weight_hh_init
        self.bias_ih_init = bias_ih_init
        self.bias_hh_init = bias_hh_init

    def init_weight_ih(self, weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        return self.weight_ih_init(weight, reference_weight)

    def init_weight_hh(self, weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        return self.weight_hh_init(weight, reference_weight)

    def init_bias_ih(self, bias: Tensor, reference_bias: Optional[Tensor] = None) -> Tensor:
        return self.bias_ih_init(bias, reference_bias)

    def init_bias_hh(self, bias: Tensor, reference_bias: Optional[Tensor] = None) -> Tensor:
        return self.bias_hh_init(bias, reference_bias)
