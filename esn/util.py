from copy import deepcopy
from typing import Tuple, List

from torch import Tensor
import torch

from esn.esn import DeepSubreservoirESN, DeepESNCell, ESNCell
from utils import math
from utils.types import Metric, SpectralCentroid, SpectralSpread


def fit_transform_DSESN(esn: DeepSubreservoirESN, input: Tensor, target: Tensor, test_input: Tensor,
                        test_target: Tensor, metric: Metric, verbose=0):
    # increase subreservoir size until it gets better on test set
    # todo optimize- can be handled with more efficient copying!-pass reference to weight matrices instead o copying them
    best_model = esn
    esn.fit(input, target)
    output = esn(test_input)
    best_output = output
    best_result = float('inf')
    new_result = metric(output.unsqueeze(-1), test_target).item()

    while new_result < best_result:
        if verbose > 0:
            print(
                f'growing improved {getattr(metric, "__name__", type(metric).__name__)} from {best_result} to {new_result}')  # todo add logging
        best_result = new_result
        best_model = deepcopy(esn)
        best_output = output
        esn.reset_hidden()
        esn.grow()
        esn.fit(input, target)
        output = esn(test_input)
        # single batch?
        new_result = metric(output.unsqueeze(-1), test_target).item()

    return best_model, best_output


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


def assign_layers(desn: DeepESNCell, input: Tensor, max_layers: int = 10, transient: int = 30,
                  tolerance: float = 0.01) -> List[SpectralCentroid]:
    in_training = True
    desn.washout(input[:transient])
    mapped_states = desn(input[transient:])
    actual_centroid, actual_spread = compute_spectral_statistics(mapped_states)
    centroids = [actual_centroid]

    # todo to jest wersja globalna, tzn jak dodanie nowej warstwy wpłynie na całą odpowiedz sieci. Ma to sens ze bedzie sie stabilizowac
    while in_training and len(desn.layers) <= max_layers:
        next_layer = ESNCell(desn.hidden_size, desn.hidden_size, desn.bias, desn.initializer, desn.activation)

        new_mapped_states = torch.cat([mapped_states, next_layer(mapped_states[:, -desn.hidden_size:])], axis=1)
        new_centroid, new_spread = compute_spectral_statistics(new_mapped_states)
        if _spectral_distance_significant(actual_centroid, new_centroid, actual_spread, tolerance):
            desn.layers.append(next_layer)
            desn.reset_hidden()
            mapped_states, actual_centroid, actual_spread = new_mapped_states, new_centroid, new_spread
        else:
            in_training = False
        centroids.append(new_centroid) #add also the one that is not in the final net
    return centroids
