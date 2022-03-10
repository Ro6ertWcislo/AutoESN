import torch
from torch import nn, Tensor

from auto_esn.esn.reservoir import activation as A
from auto_esn.esn.reservoir.activation import Activation
from auto_esn.esn.reservoir.initialization import WeightInitializer
from auto_esn.utils import math as M


class ESNCellBase(nn.Module):
    r"""
    Slightly adjusted version of torch.nn.RNNCellBase
    Main changes includes flexible initialization.
    """
    __constants__ = ['input_size', 'hidden_size', 'bias', 'weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']

    def __init__(self, input_size: int, hidden_size: int, bias: bool,
                 initializer: WeightInitializer = WeightInitializer(), num_chunks: int = 1,
                 requires_grad: bool = False, init: bool = True):
        super(ESNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.requires_grad = requires_grad
        self.bias = bias
        self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh = None, None, None, None
        if init:
            self.init_parameters(initializer)

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
        if input.size(0) != hx.size(0):  # todo add batch size??
            raise RuntimeError(
                "Input batch size {} doesn't match hidden {}".format(
                    input.size(0), hidden_label))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def get_hidden_size(self):
        return self.hidden_size

    def init_parameters(self, initializer: WeightInitializer):
        self.weight_ih = nn.Parameter(
            data=initializer.init_weight_ih(
                weight=torch.Tensor(self.hidden_size, self.input_size),
            ),
            requires_grad=self.requires_grad
        )

        self.weight_hh = nn.Parameter(
            data=initializer.init_weight_hh(
                weight=torch.Tensor(self.hidden_size, self.hidden_size),
            ),
            requires_grad=self.requires_grad
        )

        if self.bias:
            self.bias_ih = nn.Parameter(
                data=initializer.init_bias_ih(
                    bias=torch.Tensor(self.hidden_size),
                ),
                requires_grad=self.requires_grad
            )
            self.bias_hh = nn.Parameter(
                data=initializer.init_bias_hh(
                    bias=torch.Tensor(self.hidden_size),
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

    def forward(self, input: Tensor, washout=0) -> Tensor:
        if washout > 0:
            self._forward(input[:washout])
        return self._forward(input[washout:])

    def _forward(self, input: Tensor) -> Tensor:
        self.check_forward_input(input)
        if self.hx is None:
            self.hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device,
                                  requires_grad=self.requires_grad)
        self.check_forward_hidden(input, self.hx, '')

        self.map_and_activate(input)
        if input.ndim == 3:
            for i in range(input.size(0)):
                self.map_and_activate(input[i])

        return self.hx

    def map_and_activate(self, input: Tensor):
        pre_activation = M.linear(
            input, self.hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )

        self.hx = self.activation(pre_activation, prev_state=self.hx)

    def reset_hidden(self):
        self.hx = None

    def to_cuda(self):
        self.to('cuda')
        self.gpu_enabled = True


class DeepESNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias=False,
                 initializer: WeightInitializer = WeightInitializer(), num_layers: int = 1,
                 activation: Activation = 'default'):
        super().__init__()
        if type(activation) != list:
            activation = [activation] * num_layers
        else:
            activation = activation

        self.layers = [ESNCell(input_size, hidden_size, bias, initializer, activation[0])]
        if num_layers > 1:
            self.layers += [ESNCell(hidden_size, hidden_size, bias, initializer, activation[i]) for i in
                            range(1, num_layers)]
        self.gpu_enabled = False

    def forward(self, input: Tensor, washout=0) -> Tensor:
        if washout > 0:
            self._forward(input[:washout])
        return self._forward(input[washout:])

    def _forward(self, input: Tensor) -> Tensor:
        size = sum([cell.hidden_size for cell in self.layers])
        result = torch.empty((input.size(0), size), device=input.device)

        for i in range(input.size(0)):
            cell_input = input[i:i + 1]
            new_hidden_states = []
            for esn_cell in self.layers:
                cell_input = esn_cell(cell_input)
                new_hidden_states.append(cell_input)
            result[i, :] = torch.cat(new_hidden_states, axis=1)

        return result

    def get_hidden_size(self):
        return sum([l.hidden_size for l in self.layers])

    def reset_hidden(self):
        for layer in self.layers:
            layer.reset_hidden()

    def to_cuda(self):
        for layer in self.layers:
            layer.to('cuda')
        self.gpu_enabled = True


class GroupOfESNCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, groups, activation=A.self_normalizing_default(),
                 bias: bool = False, initializer: WeightInitializer = WeightInitializer()):
        super(GroupOfESNCell, self).__init__()
        num_groups = groups if type(groups) == int else len(groups)
        if type(activation) != list:
            activation = [activation] * num_groups
        else:
            activation = activation
        if type(groups) != int:
            self.groups = groups
        else:
            self.groups = [ESNCell(input_size, hidden_size, bias, initializer, activation[i]) for i in
                           range(groups)]

        self.hidden_size = hidden_size
        self.gpu_enabled = False

    def forward(self, input: Tensor, washout=0) -> Tensor:
        if washout > 0:
            self._forward(input[:washout])
        return self._forward(input[washout:])

    def _forward(self, input: Tensor) -> Tensor:
        size = self.get_hidden_size()
        result = torch.empty((input.size(0), size), device=input.device)

        for i in range(input.size(0)):
            cell_input = input[i:i + 1]
            new_hidden_states = []

            for esn_cell in self.groups:
                new_state = esn_cell(cell_input)
                new_hidden_states.append(new_state)

            result[i, :] = torch.cat(new_hidden_states, axis=1)

        return result

    def reset_hidden(self):
        for group in self.groups:
            group.reset_hidden()

    def get_hidden_size(self):
        return sum([cell.get_hidden_size() for cell in self.groups])

    def to_cuda(self):
        for group in self.groups:
            group.to('cuda')
        self.gpu_enabled = True
