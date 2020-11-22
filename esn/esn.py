import torch
from torch import nn, Tensor

from esn import activation as A
from esn.activation import Activation
from esn.initialization import WeightInitializer
from utils import math as M


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
        if input.size()[-1] != self.input_size:
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

    def forward(self, input: Tensor, no_chunk=True) -> Tensor:
        self.check_forward_input(input)
        if self.hx is None:
            self.hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device,
                                  requires_grad=self.requires_grad)
        # self.check_forward_hidden(input, self.hx, '') # todo handle both cases below

        if input.ndim == 2:
            if input.size(0) > 1 and no_chunk:
                # for now change on warining
                input = input.unsqueeze(1)
            else:
                if input.size(0) > 1:
                    print(f'ojojoj a.k.a. WARNING bitch: {input.size()}')
                self._forward(input)
        if input.ndim == 3:
            for i in range(input.size(0)):
                self._forward(input[i])

        return self.hx

    def _forward(self, input: Tensor):
        pre_activation = M.linear(
            input, self.hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
        if self.IpGain is not None and self.IpBias is not None:
            self.hx = self.activation(
                self.IpGain * pre_activation + self.IpBias,
                prev_state=self.hx
            )
        else:
            self.hx = self.activation(pre_activation, prev_state=self.hx)

    def reset_hidden(self):
        self.hx = None


class DeepESNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 initializer: WeightInitializer = WeightInitializer(), num_layers: int = 1,
                 activation: Activation = A.tanh(), include_input=False):
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
        self.include_input = include_input

    def forward(self, input: Tensor) -> Tensor:
        size =sum([cell.hidden_size for cell in self.layers])
        if self.include_input:
            size +=self.input_size

        result = torch.Tensor(input.size(0), size)

        for i in range(input.size(0)):
            cell_input = input[i]
            new_hidden_states = []
            for esn_cell in self.layers:
                cell_input = esn_cell(cell_input)
                new_hidden_states.append(cell_input)
            if self.include_input:
                result[i, :-self.input_size] = torch.cat(new_hidden_states, axis=1)
            else:
                result[i, :] = torch.cat(new_hidden_states, axis=1)

        if self.include_input:
            result[:,-self.input_size:] = input.view(input.size(0),-1)

        return result

    def washout(self, input: Tensor):
        for i in range(input.size(0)):
            cell_input = input[i]
            for esn_cell in self.layers:
                cell_input = esn_cell(cell_input)

    def reset_hidden(self):
        for layer in self.layers:
            layer.reset_hidden()


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

