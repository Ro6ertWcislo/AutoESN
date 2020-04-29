import abc

import torch.nn

from utils.math import spectral_norm


# todo refactor from decorators into builder-like flow?

class ConcreteWeightInitializer(abc.ABC):

    @abc.abstractmethod
    def __call__(self, weight, hidden_size):
        pass


class Scale(ConcreteWeightInitializer):
    def __init__(self, initializer: ConcreteWeightInitializer, factor: float = 1):
        self.inner_initializer = initializer
        self.factor = factor

    def __call__(self, weight, hidden_size):
        self.inner_initializer(weight, hidden_size)
        with torch.no_grad():
            weight[:] = weight * self.factor


class Normalize(ConcreteWeightInitializer):
    def __init__(self, initializer: ConcreteWeightInitializer):
        self.inner_initializer = initializer

    def __call__(self, weight, hidden_size):
        self.inner_initializer(weight, hidden_size)
        with torch.no_grad():
            weight[:] = spectral_norm(weight)


class Sparse(ConcreteWeightInitializer):
    def __init__(self, initializer: ConcreteWeightInitializer, density: float = 1):
        self.inner_initializer = initializer
        self.density = density

    def __call__(self, weight, hidden_size):
        self.inner_initializer(weight, hidden_size)
        zero_mask = torch.FloatTensor(hidden_size, hidden_size).uniform_() > (1 - self.density)
        with torch.no_grad():
            weight[:] = weight * zero_mask


class Uniform(ConcreteWeightInitializer):
    def __init__(self, min: int = -1, max: int = 1):
        self.min = min
        self.max = max

    def __call__(self, weight, hidden_size):
        torch.nn.init.uniform_(weight, self.min, self.max)


class Xavier(ConcreteWeightInitializer):
    def __call__(self, weight, hidden_size):
        torch.nn.init.xavier_uniform_(weight)


class SpectralNoisy(ConcreteWeightInitializer):
    def __init__(self, spectral_radius: float = 0.9, noise_magnitude: float = 0.2):
        self.spectral_radius = spectral_radius
        self.noise_magnitude = noise_magnitude

    def __call__(self, weight, hidden_size):
        def scale(tensor):
            return (tensor / hidden_size) * self.spectral_radius

        w_hh = scale(torch.ones_like(weight))
        noise = scale(torch.rand_like(w_hh)) * self.noise_magnitude
        w_hh = w_hh + (noise - torch.mean(noise))
        with torch.no_grad():
            weight[:] = w_hh


class DefaultHidden(ConcreteWeightInitializer):
    def __init__(self, density=0.1, spectral_radius=0.9):
        self.stacked_initializer = \
            Scale(
                Normalize(
                    Sparse(
                        Xavier(), density=density
                    )
                ), factor=spectral_radius)

    def __call__(self, weight, hidden):
        self.stacked_initializer(weight, hidden)


class WeightInitializer(object):
    def __init__(self,
                 weight_ih_init: ConcreteWeightInitializer = Uniform(),
                 weight_hh_init: ConcreteWeightInitializer = DefaultHidden(),
                 bias_ih_init: ConcreteWeightInitializer = Uniform(),
                 bias_hh_init: ConcreteWeightInitializer = Uniform()):
        self.weight_ih_init = weight_ih_init
        self.weight_hh_init = weight_hh_init
        self.bias_ih_init = bias_ih_init
        self.bias_hh_init = bias_hh_init

    def init_weight_ih(self, weight, hidden_size):
        self.weight_ih_init(weight, hidden_size)

    def init_weight_hh(self, weight, hidden_size):
        self.weight_hh_init(weight, hidden_size)

    def init_bias_ih(self, bias, hidden_size):
        self.bias_ih_init(bias, hidden_size)

    def init_bias_hh(self, bias, hidden_size):
        self.bias_hh_init(bias, hidden_size)
