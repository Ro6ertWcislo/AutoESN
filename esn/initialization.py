from typing import List, Optional

import enum
import torch.nn
from torch import Tensor

from utils.math import spectral_normalize
from utils.types import Initializer


# todo clean?
def _scale(factor: float = 1.0) -> Initializer:
    def __scale(weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        return weight * factor

    return __scale


def _spectral_normalize() -> Initializer:
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

    def spectral_normalize(self):
        self.initializers.append(_spectral_normalize())
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


def dense(min_val=-1, max_val=1, spectral_radius=0.9) -> Initializer:
    return CompositeInitializer() \
        .uniform(min_val=min_val, max_val=max_val) \
        .spectral_normalize() \
        .scale(factor=spectral_radius)


class GrowingType(enum.Enum):
    Reservoir = 1
    Input = 2
    Bias = 3
    InputPad = 4


class SubreservoirInitializer(object):
    # todo generate without svd
    def __init__(self, growing_type: GrowingType, subreservoir_size: int = 10,
                 initializer: CompositeInitializer = dense()):
        self.growing_type = growing_type
        self.initalizer = initializer
        self.subreservoir_size = subreservoir_size

    def __call__(self, weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        if reference_weight is None:
            return self.initalizer(weight)
        else:
            if self.growing_type == GrowingType.Reservoir:
                result = torch.zeros_like(weight)
                next_subreservoir = self.initalizer(torch.Tensor(self.subreservoir_size, self.subreservoir_size))
                result[:reference_weight.size(0), :reference_weight.size(1)] = reference_weight
                result[reference_weight.size(0):, reference_weight.size(1):] = next_subreservoir
                return result

            elif self.growing_type == GrowingType.Input:
                input_weight_extension = self.initalizer(torch.Tensor(self.subreservoir_size, reference_weight.size(1)))
                return torch.cat([reference_weight, input_weight_extension], axis=0)

            elif self.growing_type == GrowingType.InputPad:  # todo refactor
                input_weight_extension = self.initalizer(torch.Tensor(weight.size(1),self.subreservoir_size))
                return torch.cat([reference_weight, input_weight_extension], axis=1)

            elif self.growing_type == GrowingType.Bias:
                bias_extension = self.initalizer(torch.Tensor(self.subreservoir_size))
                return torch.cat([reference_weight, bias_extension])


def uniform(min_val=-1, max_val=1) -> Initializer:
    return CompositeInitializer().uniform(min_val=min_val, max_val=max_val)


def default_hidden(density=0.1, spectral_radius=0.9):
    return CompositeInitializer() \
        .xavier_uniform() \
        .sparse(density=density) \
        .spectral_normalize() \
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


class SubreservoirWeightInitializer(WeightInitializer):
    def __init__(self,
                 subreservoir_size: int = 10,
                 weight_ih_init: Initializer = uniform(),
                 weight_hh_init: Initializer = dense(),
                 bias_ih_init: Initializer = uniform(),
                 bias_hh_init: Initializer = uniform()):
        super().__init__(
            SubreservoirInitializer(GrowingType.Input, subreservoir_size, weight_ih_init),
            SubreservoirInitializer(GrowingType.Reservoir, subreservoir_size, weight_hh_init),
            SubreservoirInitializer(GrowingType.Bias, subreservoir_size, bias_ih_init),
            SubreservoirInitializer(GrowingType.Bias, subreservoir_size, bias_hh_init)
        )
        self.subreservoir_size = subreservoir_size
        self.weight_ih_pad = SubreservoirInitializer(GrowingType.InputPad, subreservoir_size,
                                                     weight_ih_init)  # todo moze calosc od razu? ma to sens?

    def init_weight_ih_pad(self, weight: Tensor, reference_weight: Optional[Tensor] = None) -> Tensor:
        return self.weight_ih_pad(weight, reference_weight)
