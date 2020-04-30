import torch.nn

from utils.math import spectral_norm
from utils.types import Initializer


class Scale(object):
    def __init__(self, initializer: Initializer, factor: float = 1):
        self.inner_initializer = initializer
        self.factor = factor

    def __call__(self, weight, hidden_size):
        self.inner_initializer(weight, hidden_size)
        with torch.no_grad():
            weight[:] = weight * self.factor


class Normalize(object):
    def __init__(self, initializer: Initializer):
        self.inner_initializer = initializer

    def __call__(self, weight, hidden_size):
        self.inner_initializer(weight, hidden_size)
        with torch.no_grad():
            weight[:] = spectral_norm(weight)


class Sparse(object):
    def __init__(self, initializer: Initializer, density: float = 1):
        self.inner_initializer = initializer
        self.density = density

    def __call__(self, weight, hidden_size):
        self.inner_initializer(weight, hidden_size)
        zero_mask = torch.FloatTensor(hidden_size, hidden_size).uniform_() > (1 - self.density)
        with torch.no_grad():
            weight[:] = weight * zero_mask


class Uniform(object):
    def __init__(self, min: int = -1, max: int = 1):
        self.min = min
        self.max = max

    def __call__(self, weight, hidden_size):
        torch.nn.init.uniform_(weight, self.min, self.max)


class Xavier(object):
    def __call__(self, weight, hidden_size):
        torch.nn.init.xavier_uniform_(weight)


class SpectralNoisy(object):
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


class DefaultHidden(object):
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
                 weight_ih_init: Initializer = Uniform(),
                 weight_hh_init: Initializer = DefaultHidden(),
                 bias_ih_init: Initializer = Uniform(),
                 bias_hh_init: Initializer = Uniform()):
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
