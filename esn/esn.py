from typing import List

import torch
from torch import nn, Tensor

from esn import activation as A
from esn.activation import Activation
from esn.initialization import WeightInitializer, SubreservoirWeightInitializer
from utils import math as M
from utils.types import SpectralCentroid


class ESNCellBase(nn.Module):
    r"""
    Slightly adjusted version of torch.nn.RNNCellBase
    Main changes includes flexible initialization.
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']

    def __init__(self, input_size: int, hidden_size: int, bias: bool,
                 initializer: WeightInitializer = WeightInitializer(), num_chunks: int = 1,
                 requires_grad: bool = False):
        super(ESNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.requires_grad = requires_grad
        self.bias = bias
        self.initializer = initializer
        self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh = None, None, None, None
        self.IpGain, self.IpBias = None, None
        self.init_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input: Tensor):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = ''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def init_parameters(self):
        # todo  change to register params and receive just size?
        # todo chunks, what about chunks
        self.weight_ih = nn.Parameter(
            data=self.initializer.init_weight_ih(
                weight=torch.Tensor(self.hidden_size, self.input_size),
                reference_weight=self.weight_ih
            ),
            requires_grad=self.requires_grad
        )

        self.weight_hh = nn.Parameter(
            data=self.initializer.init_weight_hh(
                weight=torch.Tensor(self.hidden_size, self.hidden_size),
                reference_weight=self.weight_hh
            ),
            requires_grad=self.requires_grad
        )

        if self.bias:
            self.bias_ih = nn.Parameter(
                data=self.initializer.init_bias_ih(
                    bias=torch.Tensor(self.hidden_size),
                    reference_bias=self.bias_ih
                ),
                requires_grad=self.requires_grad
            )
            self.bias_hh = nn.Parameter(
                data=self.initializer.init_bias_hh(
                    bias=torch.Tensor(self.hidden_size),
                    reference_bias=self.bias_hh
                ),
                requires_grad=self.requires_grad
            )


class ESNCell(ESNCellBase):
    __constants__ = ['input_size', 'hidden_size', 'bias', 'activation']

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True,
                 initializer: WeightInitializer = WeightInitializer(),
                 activation: Activation = A.tanh(), num_chunks: int = 1, requires_grad: bool = False):
        super(ESNCell, self).__init__(input_size, hidden_size, bias, initializer=initializer, num_chunks=num_chunks,
                                      requires_grad=requires_grad)
        self.requires_grad = requires_grad
        self.activation = activation
        self.hx = None

    def forward(self, input: Tensor) -> Tensor:
        self.check_forward_input(input)
        if self.hx is None:
            self.hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device,
                                  requires_grad=self.requires_grad)
        self.check_forward_hidden(input, self.hx, '')
        pre_activation = M.linear(
            input, self.hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
        if self.IpGain is not None and self.IpBias is not None:
            self.hx = self.activation(self.IpGain * pre_activation + self.IpBias, prev_state=self.hx)
        else:
            self.hx = self.activation(pre_activation, prev_state=self.hx)

        return self.hx

    def reset_hidden(self):
        self.hx = None

    def intristic_plasticity_pretrain(self, input: Tensor, epochs: int = 10, mean: float = 0., variance: float = 1.,
                                      learning_rate: float = 10e-4):
        self.IpGain = torch.ones(1, self.hidden_size)  # todo stay on if broadcasting, change on chunk if needed
        self.IpBias = torch.zeros(1, self.hidden_size)

        for i in range(epochs):
            print(f'starting learning for epoch {i+1} out of {epochs}. Dims: input: {input.size()}. cell: in:{self.input_size}, hidden: {self.hidden_size}.')
            self.hx = torch.zeros(1, self.hidden_size, dtype=input.dtype, device=input.device,
                                  requires_grad=self.requires_grad)
            for i in range(input.size(0)):
                cell_input = input[i]
                pre_activation = M.linear(
                    cell_input, self.hx,
                    self.weight_ih, self.weight_hh,
                    self.bias_ih, self.bias_hh,
                )
                post_activation = self.activation(pre_activation, prev_state=self.hx)
                bias_delta = - learning_rate * (-(mean / variance) + (post_activation / variance) * (
                        2 * variance + 1 - post_activation ** 2 + mean * post_activation))
                self.IpBias = self.IpBias + bias_delta
                self.IpGain = self.IpGain + (
                        torch.ones_like(self.IpGain) * learning_rate) / self.IpGain + bias_delta * pre_activation
                self.hx = post_activation
        self.reset_hidden()


class SubreservoirCell(ESNCell):
    def __init__(self, input_size: int, bias: bool = True,
                 initializer: SubreservoirWeightInitializer = SubreservoirWeightInitializer(),
                 activation: Activation = A.tanh(), requires_grad: bool = False, input_cell=False):
        super(SubreservoirCell, self).__init__(input_size,
                                               initializer.subreservoir_size,
                                               bias,
                                               initializer=initializer,
                                               num_chunks=1,
                                               activation=activation,
                                               requires_grad=requires_grad)
        self.input_cell = input_cell

    def fix_input_layer(self, previous_cell: 'SubreservoirCell' = None):
        # after growing previous layer the dimensions will not match
        if previous_cell is not None:
            self.weight_ih = nn.Parameter(
                data=self.initializer.init_weight_ih_pad(
                    weight=torch.Tensor(self.hidden_size, previous_cell.weight_hh.size(1)),
                    reference_weight=self.weight_ih
                ),
                requires_grad=self.requires_grad
            )
            self.input_size += self.initializer.subreservoir_size

    def grow(self):
        self.hidden_size += self.initializer.subreservoir_size
        # if not self.input_cell:
        #     self.input_size += self.initializer.subreservoir_size
        self.init_parameters()


# todo handle different layer sizes
class DeepESNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 initializer: WeightInitializer = WeightInitializer(), num_layers: int = 1,
                 activation: Activation = A.tanh()):
        super().__init__()
        self.activation = activation
        if num_layers > 0:
            self.layers = [ESNCell(input_size, hidden_size, bias, initializer, activation)] + \
                          [ESNCell(hidden_size, hidden_size, bias, initializer, activation) for _ in
                           range(1, num_layers)]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.initializer = initializer
        self.activation = activation

    def forward(self, input: Tensor) -> Tensor:
        result = torch.Tensor(input.size(0), len(self.layers) * self.hidden_size)

        for i in range(input.size(0)):
            cell_input = input[i]
            new_hidden_states = []
            for esn_cell in self.layers:
                cell_input = esn_cell(cell_input)
                new_hidden_states.append(cell_input)
            result[i, :] = torch.cat(new_hidden_states, axis=1)  # todo check for multivariete

        return result

    def intristic_plasticity_pretrain(self, input, epochs: int = 10, mean: float = 0., variance: float = 1.,
                    learning_rate: float = 10e-4):
        cell_input = input[:,0,:]
        for num,esn_cell in enumerate(self.layers):
            print(f'starting layer: {num+1} out of {len(self.layers)}')
            esn_cell.intristic_plasticity_pretrain(cell_input,epochs=epochs,mean=mean,variance=variance,learning_rate=learning_rate)
            cell_input = [esn_cell(cell_input[t].unsqueeze(0)) for t in range(cell_input.size(0))]
            cell_input  = torch.cat(cell_input,axis=0)

    def washout(self, input: Tensor):
        for i in range(input.size(0)):
            cell_input = input[i]
            for esn_cell in self.layers:
                cell_input = esn_cell(cell_input)

    def reset_hidden(self):
        for layer in self.layers:
            layer.reset_hidden()

    def assign_layers(self, input: Tensor, max_layers: int = 10, transient: int = 30,
                      tolerance: float = 0.01) -> List[SpectralCentroid]:
        in_training = True
        self.washout(input[:transient])
        mapped_states = self.forward(input[transient:])
        actual_centroid, actual_spread = M.compute_spectral_statistics(mapped_states)
        centroids = [actual_centroid]

        # todo to jest wersja globalna, tzn jak dodanie nowej warstwy wpłynie na całą odpowiedz sieci. Ma to sens ze bedzie sie stabilizowac
        while in_training and len(self.layers) <= max_layers:
            next_layer = ESNCell(self.hidden_size, self.hidden_size, self.bias, self.initializer, self.activation)

            new_mapped_states = torch.cat([mapped_states, next_layer(mapped_states[:, -self.hidden_size:])], axis=1)
            new_centroid, new_spread = M.compute_spectral_statistics(new_mapped_states)
            if M._spectral_distance_significant(actual_centroid, new_centroid, actual_spread, tolerance):
                self.layers.append(next_layer)
                self.reset_hidden()
                mapped_states, actual_centroid, actual_spread = new_mapped_states, new_centroid, new_spread
            else:
                in_training = False
            centroids.append(new_centroid)  # add also the one that is not in the final net
        return centroids


class DeepSubreservoirCell(DeepESNCell):
    def __init__(self,
                 input_size: int,
                 bias=True,
                 initializer: SubreservoirWeightInitializer = SubreservoirWeightInitializer(),
                 num_layers: int = 1,
                 activation: Activation = A.tanh()):
        super().__init__(input_size, None, bias, initializer, 0, activation)
        self.layers = [SubreservoirCell(input_size, bias, initializer, activation, input_cell=True)] + \
                      [SubreservoirCell(initializer.subreservoir_size, bias, initializer, activation) for _ in
                       range(1, num_layers)]
        self.subreservoir_size = initializer.subreservoir_size
        self.hidden_size = initializer.subreservoir_size

    def grow(self):
        self.hidden_size += self.subreservoir_size
        for i in range(len(self.layers)):
            self.layers[i].grow()
            if i > 0:
                self.layers[i].fix_input_layer(self.layers[i - 1])


class SVDReadout(nn.Module):
    def __init__(self, total_hidden_size: int, output_dim: int, regularization: float = 1.):
        super().__init__()
        self.readout = nn.Linear(total_hidden_size, output_dim)
        self.regularization = regularization

    def forward(self, input: Tensor) -> Tensor:
        return self.readout(input)

    def fit(self, input: Tensor, target: Tensor):
        X = torch.ones(input.size(0), 1 + input.size(1), device=target.device)
        X[:, :-1] = input
        W = self._solve_svd(X, target, self.regularization)
        self.readout.bias = nn.Parameter(W[:, -1], requires_grad=False)
        self.readout.weight = nn.Parameter(W[:, :-1], requires_grad=False)

    def _solve_svd(self, X: Tensor, y: Tensor, alpha: float) -> Tensor:
        # implementation taken from scikit-learn
        y = y[:, 0, :]  # ignore batch
        U, s, V = torch.svd(X)
        idx = s > 1e-15  # same default value as scipy.linalg.pinv
        s_nnz = s[idx][:, None]
        UTy = U.T @ y
        d = torch.zeros(s.size(0), 1, device=X.device)
        d[idx] = s_nnz / (s_nnz ** 2 + alpha)
        d_UT_y = d * UTy

        return (V @ d_UT_y).T


class ESNBase(nn.Module):
    def __init__(self, reservoir: nn.Module, readout: nn.Module,
                 transient: int = 30):  # todo typy poprawic. module jest zle
        super(ESNBase, self).__init__()
        self.transient = transient
        self.initial_state = True
        self.reservoir = reservoir
        self.readout = readout

    def fit(self, input: Tensor, target: Tensor):
        if self.initial_state:
            self.initial_state = False
            self.reservoir.washout(input[:self.transient])
            mapped_input = self.reservoir(input[self.transient:])
            self.readout.fit(mapped_input, target[self.transient:])
        else:
            mapped_input = self.reservoir(input)
            self.readout.fit(mapped_input, target)

    def forward(self, input: Tensor) -> Tensor:
        self.initial_state = False
        mapped_input = self.reservoir(input)

        return self.readout(mapped_input)

    def reset_hidden(self):
        self.initial_state = True
        self.reservoir.reset_hidden()


class DeepESN(ESNBase):
    def __init__(self, input_size: int, hidden_size: int, output_dim: int = 1, bias: bool = True,
                 initializer: WeightInitializer = None, num_layers=1, activation: Activation = A.tanh(),
                 transient: int = 30, reglarization: float = 1.):
        super().__init__(
            reservoir=DeepESNCell(input_size, hidden_size, bias, initializer, num_layers, activation),
            readout=SVDReadout(hidden_size * num_layers, output_dim, regularization=reglarization),
            transient=transient)


class DeepSubreservoirESN(ESNBase):
    def __init__(self, input_size: int, output_dim: int = 1, bias: bool = True,
                 initializer: SubreservoirWeightInitializer = SubreservoirWeightInitializer(), num_layers=1,
                 activation: Activation = A.tanh(),
                 transient: int = 30, regularization: float = 1.):
        super().__init__(
            reservoir=DeepSubreservoirCell(input_size, bias, initializer, num_layers, activation),
            readout=SVDReadout(initializer.subreservoir_size * num_layers, output_dim, regularization=regularization),
            transient=transient)
        self.output_dim = output_dim
        self.regularization = regularization
        self.hidden_size = self.reservoir.hidden_size

    def grow(self):
        self.reservoir.grow()
        self.hidden_size = self.reservoir.hidden_size
        self.readout = SVDReadout(self.hidden_size * len(self.reservoir.layers), self.output_dim, self.regularization)
