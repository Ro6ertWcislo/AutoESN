import math
from random import sample
from typing import List, Union

import networkx as nx
import numpy as np
import torch.nn
from networkx import watts_strogatz_graph, adjacency_matrix
from torch import Tensor

from auto_esn.esn.reservoir.util import get_regular_graph_mask, get_star_graph_mask, set_all_seeds

# todo clean?
from auto_esn.utils.math import spectral_normalize
from auto_esn.utils.types import Initializer


def _scale(factor: float = 1.0) -> Initializer:
    def __scale(weight: Tensor) -> Tensor:
        return weight * factor

    return __scale


def wrap(fun, *args, **kwargs) -> Initializer:
    def wrapped(weight: Tensor) -> Tensor:
        return fun(weight, *args, **kwargs)

    return wrapped


def _spectral_normalize() -> Initializer:
    def __normalize(weight: Tensor) -> Tensor:
        return spectral_normalize(weight)

    return __normalize


def _sparse(density: float = 1.0) -> Initializer:
    def __sparse(weight: Tensor) -> Tensor:
        zero_mask = torch.empty_like(weight).uniform_() > (1 - density)
        return weight * zero_mask

    return __sparse


def _uniform(min_val: float = -1, max_val: float = 1) -> Initializer:
    def __uniform(weight: Tensor) -> Tensor:
        new_weight = torch.empty_like(weight)
        torch.nn.init.uniform_(new_weight, min_val, max_val)
        return new_weight

    return __uniform


def _regular_graph(degree_or_density: Union[int, float] = 3) -> Initializer:
    def __regular_graph(weight: Tensor) -> Tensor:
        nodes = weight.size(0)
        if type(degree_or_density) == float:
            # degree * nodes == density * nodes^2
            degree = max(int(degree_or_density * nodes), 1)
        else:
            degree = degree_or_density

        if degree * nodes % 2 == 1:
            nodes += 1  # networkx needs degree * nodes to be even to generate the graph
        graph_mask = get_regular_graph_mask(degree=degree, nodes=nodes)
        return weight * graph_mask

    return __regular_graph


def _watts_strogatz(neighbours: int = 10, rewire_proba: float = 0.05) -> Initializer:
    def __watts_strogatz(weight: Tensor) -> Tensor:
        nodes = weight.size(0)
        G = watts_strogatz_graph(n=nodes, k=neighbours, p=rewire_proba)
        graph_mask = torch.from_numpy(adjacency_matrix(G).toarray())
        return weight * graph_mask

    return __watts_strogatz


def _star_graph(stars_or_density: Union[int, float] = 3) -> Initializer:
    def __star_graph(weight: Tensor) -> Tensor:
        nodes = weight.size(0)
        if type(stars_or_density) == float:
            # 2 * stars * nodes == density * nodes^2
            stars = max(int((stars_or_density * nodes) // 2), 1)
        else:
            stars = stars_or_density
        graph_mask = get_star_graph_mask(size=nodes, stars=stars)
        return weight * graph_mask

    return __star_graph


def random_zero(weight: Tensor, zero_density: float = 0.05, symmetric: bool = True):
    """
    zeros  zero_density values and then sets other random_density to some uniform random number
    It doesnt guarantee
    """
    size = weight.size(0)

    total_samples = int(size ** 2 * zero_density)
    if symmetric:
        total_samples = total_samples // 2

    if zero_density > 0:
        if symmetric:
            non_zero_indices = ((weight * torch.triu(torch.ones_like(weight), diagonal=1)) != 0).nonzero()
        else:
            non_zero_indices = (weight != 0).nonzero()
        if non_zero_indices.size(0) < total_samples:
            raise RuntimeError("There are less non zeros than values you want to zero!")
        sampling = torch.randperm(non_zero_indices.size(0))[:total_samples]
        coords = non_zero_indices[sampling].T
        zero_mask = torch.ones_like(weight)
        zero_mask[coords[0], coords[1]] = 0
        if symmetric:
            zero_mask[coords[1], coords[0]] = 0
        weight = weight * zero_mask

    return weight


def random_add(weight: Tensor, additional_density: float = 0.05, symmetric: bool = True):
    """
    zeros  zero_density values and then sets other random_density to some uniform random number
    It doesnt guarantee
    """
    size = weight.size(0)
    total_samples = int(size ** 2 * additional_density)
    if symmetric:
        total_samples = total_samples // 2
    if additional_density > 0:
        if symmetric:
            zero_indices = ((weight + torch.tril(torch.ones_like(weight), diagonal=0)) == 0).nonzero()
        else:
            zero_indices = (weight == 0).nonzero()
        if zero_indices.size(0) < total_samples:
            raise RuntimeError("There are less zeros than values you want to add!")
        sampling = torch.randperm(zero_indices.size(0))[:total_samples]
        coords = zero_indices[sampling].T
        rand_mask = torch.rand_like(weight)
        zero_mask = torch.zeros_like(weight)
        zero_mask[coords[0], coords[1]] = 1
        if symmetric:
            zero_mask[coords[1], coords[0]] = 1
        weight = weight + (zero_mask * rand_mask)

    return weight


def separate(weight: Tensor, input_nodes: int = 200, output_nodes=200):
    size = weight.size(0)
    all_indices = sample(range(size), input_nodes + output_nodes)
    input_indices = set(all_indices[:input_nodes])
    output_indices = set(all_indices[input_nodes:])
    non_output_indices = set(range(size)) - output_indices

    for index in input_indices:
        connections = weight[index].nonzero()  # nonzero() returns one-element tuple
        connections = set(connections.reshape(-1).numpy())
        conflicts = output_indices.intersection(connections)
        new_indices = sample(tuple(non_output_indices), len(conflicts))
        for conflict, new_index in zip(conflicts, new_indices):
            weight[index, new_index] = weight[index, conflict]
            weight[index, conflict] = 0.0

    return weight


def loopify(weight: Tensor, percentage_of_loops: float = 1.0):
    """
    sets $percentage_of_loops values on diagonal to non-zero values
    """
    size = weight.size(0)
    total_samples = int(percentage_of_loops * size)
    zero_diag_mask = torch.tensor([[1]]) - torch.eye(size)
    weight = weight * zero_diag_mask
    if total_samples > 0:
        sampling = torch.randperm(size)[:total_samples]
        rand_mask = torch.rand_like(weight)
        zero_diag_mask[sampling, sampling] = 1
        weight = weight + (rand_mask * zero_diag_mask)
    return weight


def isomorph(weight: Tensor):
    permutation = torch.randperm(weight.size(0))
    permutation_matrix = torch.eye(weight.size(0))
    permutation_matrix = permutation_matrix[:, permutation]
    return weight @ permutation_matrix


def configuration_model(weight: Tensor, configuration: List[int]):
    G = nx.configuration_model(configuration)
    mask = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
    return weight * torch.from_numpy(mask)


def expander(weight: Tensor, expander_name: str = "paley"):
    expanders = {
        "mgg": lambda n: nx.margulis_gabber_galil_graph(int(math.sqrt(n))),
        # margulis_gabber_galil_graph - weight size must have integer square root
        "chordal": nx.chordal_cycle_graph,  # chordal_cycle_graph - weight size must be prime number
        "paley": nx.paley_graph  # paley_graph  - weight size must be prime number

    }
    G = expanders[expander_name](weight.size(0))
    mask = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
    return weight * torch.from_numpy(mask)


def subreservoir(weight: Tensor, k=3):
    """
    size of weight must devide by k
    """
    subres_size = weight.size(0) // k
    mask = np.zeros_like(weight)
    for i in range(k):
        mask[i * subres_size: (i + 1) * subres_size, i * subres_size: (i + 1) * subres_size] = torch.ones(subres_size,
                                                                                                          subres_size)
    return weight * mask


def init_seed(weight: Tensor, seed: int = 42):
    set_all_seeds(seed)
    return weight


def _xavier_uniform() -> Initializer:
    def __uniform(weight: Tensor) -> Tensor:
        new_weight = torch.empty_like(weight)
        torch.nn.init.xavier_uniform_(new_weight)
        return new_weight

    return __uniform


def _spectral_noisy(spectral_radius=0.9, noise_magnitude=0.2) -> Initializer:
    def __spectral_noisy(weight: Tensor) -> Tensor:
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

    def regular_graph(self, degree_or_density: Union[float, int] = 3):
        self.initializers.append(_regular_graph(degree_or_density))
        return self

    def watts_strogatz(self, neighbours: int = 10, rewire_proba: float = 0.05):
        self.initializers.append(_watts_strogatz(neighbours, rewire_proba))
        return self

    def star_graph(self, stars_or_density: Union[float, int] = 3):
        self.initializers.append(_star_graph(stars_or_density))
        return self

    def random_zero(self, zero_density: float = 0.05, symmetric: bool = True):
        self.initializers.append(wrap(random_zero, zero_density, symmetric))
        return self

    def random_add(self, additional_density: float = 0.05, symmetric: bool = True):
        self.initializers.append(wrap(random_add, additional_density, symmetric))
        return self

    def loopify(self, percentage_of_loops: float = 1.0):
        self.initializers.append(wrap(loopify, percentage_of_loops))
        return self

    def separate(self, input_nodes: int = 200, output_nodes: int = 200):
        self.initializers.append(wrap(separate, input_nodes, output_nodes))
        return self

    def configuration_model(self, configuration: List[int]):
        self.initializers.append(wrap(configuration_model, configuration))
        return self

    def expander(self, expander_name: str = "paley"):
        self.initializers.append(wrap(expander, expander_name))
        return self

    def subreservoir(self, k=3):
        self.initializers.append(wrap(subreservoir, k))
        return self

    def isomorph(self):
        self.initializers.append(wrap(isomorph))
        return self

    def with_seed(self, seed):
        set_all_seeds(seed)
        return self

    def init_seed(self, seed):
        self.initializers.append(wrap(init_seed, seed))
        return self

    def __call__(self, weight: Tensor) -> Tensor:
        for initializer in self.initializers:
            weight = initializer(weight)
        return weight


def dense(min_val=-1, max_val=1, spectral_radius=0.9) -> Initializer:
    return CompositeInitializer() \
        .uniform(min_val=min_val, max_val=max_val) \
        .spectral_normalize() \
        .scale(factor=spectral_radius)


def uniform(min_val=-1, max_val=1) -> Initializer:
    return CompositeInitializer().uniform(min_val=min_val, max_val=max_val)


def default_hidden(density=0.1, spectral_radius=0.9):
    return CompositeInitializer() \
        .uniform() \
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

    def init_weight_ih(self, weight: Tensor) -> Tensor:
        return self.weight_ih_init(weight)

    def init_weight_hh(self, weight: Tensor) -> Tensor:
        return self.weight_hh_init(weight)

    def init_bias_ih(self, bias: Tensor) -> Tensor:
        return self.bias_ih_init(bias)

    def init_bias_hh(self, bias: Tensor) -> Tensor:
        return self.bias_hh_init(bias)
