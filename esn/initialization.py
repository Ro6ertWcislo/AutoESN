import torch.nn
import math


def default_init(weight, hidden_size):
    stdv = 1.0 / math.sqrt(hidden_size)
    torch.nn.init.uniform_(weight, -stdv, stdv)










class SpectralNoisyInitializer(object):
    def __init__(self, spectral_radius=0.9):
        self.spectral_radius = spectral_radius

    def __call__(self, weight,hidden_size):
        def scale(tensor):
            return (tensor / hidden_size) * self.spectral_radius

        w_hh = scale(torch.ones_like(weight))
        noise = scale(torch.rand_like(w_hh)) * 0.7
        w_hh = w_hh + (noise - torch.mean(noise))
        with torch.no_grad():
            weight[:] = w_hh


