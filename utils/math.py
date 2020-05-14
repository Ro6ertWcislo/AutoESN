from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from utils.types import SpectralCentroid, SpectralSpread


def spectral_normalize(tensor: Tensor) -> Tensor:
    u, s, v = torch.svd(tensor, compute_uv=False)
    norm = torch.max(s)
    return tensor / norm


def eigen_norm(tensor: Tensor) -> Tensor:
    abs_eigs = (torch.eig(tensor)[0] ** 2).sum(1).sqrt()
    return tensor / torch.max(abs_eigs)


def linear(input: Tensor, hx: Tensor, weight_ih: Tensor, weight_hh: Tensor, bias_ih: Optional[Tensor],
           bias_hh: Optional[Tensor]) -> Tensor:
    return F.linear(input, weight_ih, bias_ih) + F.linear(hx, weight_hh, bias_hh)


def leaky(hx_prev: Tensor, hx_next: Tensor, leaky_rate: float = 1.) -> Tensor:
    return (1. - leaky_rate) * hx_prev + leaky_rate * hx_next


def FFT(mapped_states: Tensor) -> Tuple[Tensor, Tensor]:
    timesteps = mapped_states.size(1)
    frequencies = torch.zeros(mapped_states.size(0), timesteps // 2)
    for i in range(mapped_states.size(0)):
        signal = mapped_states[i]
        signal_with_complex = torch.cat([signal.unsqueeze(-1), torch.zeros(timesteps, 1)], axis=1)
        comps = torch.fft(signal_with_complex, signal_ndim=1)
        frequencies[i, :] = torch.norm(comps[:timesteps // 2], p=2, dim=1)
    avg_magnitudes = torch.mean(frequencies, dim=0)
    return avg_magnitudes, torch.linspace(1, timesteps // 2, timesteps // 2) / timesteps


def _spectral_centroid(magnitudes: Tensor, frequencies: Tensor) -> SpectralCentroid:
    return (torch.sum(magnitudes * frequencies) / torch.sum(magnitudes)).item()


def _spectral_spread(magnitudes: Tensor, frequencies: Tensor, centroid: float) -> SpectralSpread:
    return (torch.sqrt(torch.sum(magnitudes * ((frequencies - centroid) ** 2)) / torch.sum(magnitudes))).item()


def _spectral_distance_significant(centroid1: float, centroid2: float, spread1: float, tolerance: float) -> bool:
    return abs(centroid1 - centroid2) > spread1 * tolerance


def compute_spectral_statistics(mapped_states: Tensor) -> Tuple[SpectralCentroid, SpectralSpread]:
    magnitudes, frequencies = FFT(mapped_states)
    centroid = _spectral_centroid(magnitudes, frequencies)
    spread = _spectral_spread(magnitudes, frequencies, centroid)

    return centroid, spread
