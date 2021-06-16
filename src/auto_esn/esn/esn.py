from torch import nn, Tensor

from auto_esn.esn.readout.svr_readout import SVDReadout
from auto_esn.esn.reservoir import activation as A
from auto_esn.esn.reservoir.activation import Activation
from auto_esn.esn.reservoir.cell import DeepESNCell, GroupOfESNCell
from auto_esn.esn.reservoir.initialization import WeightInitializer


class ESNBase(nn.Module):
    def __init__(self, reservoir: nn.Module, readout: nn.Module,
                 transient: int = 30):
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

    def to_cuda(self):
        self.reservoir.to_cuda()
        self.readout.to_cuda()


class DeepESN(ESNBase):
    def __init__(self, input_size: int = 1, hidden_size: int = 500, output_dim: int = 1, bias: bool = False,
                 initializer: WeightInitializer = WeightInitializer(), num_layers=2,
                 activation=A.self_normalizing_default(), transient: int = 30, regularization: float = 1.,
                 leaky_rate=1.0, act_radius=100,
                 act_grow='decr'):
        super().__init__(
            reservoir=DeepESNCell(input_size, hidden_size, bias, initializer, num_layers, activation),
            readout=SVDReadout(hidden_size * num_layers, output_dim, regularization=regularization),
            transient=transient)


class GroupOfESN(ESNBase):
    def __init__(self, input_size: int = 1, hidden_size: int = 250, output_dim: int = 1, bias: bool = False,
                 initializer: WeightInitializer = WeightInitializer(), groups=4,
                 activation: Activation = A.self_normalizing_default(), transient: int = 30,
                 regularization: float = 1.):
        super().__init__(
            reservoir=GroupOfESNCell(input_size, hidden_size, groups, activation, bias, initializer),
            readout=SVDReadout(hidden_size * groups, output_dim, regularization=regularization),
            transient=transient)


class FlexDeepESN(ESNBase):
    def __init__(self, readout, input_size: int = 1, hidden_size: int = 500, bias: bool = False,
                 initializer: WeightInitializer = WeightInitializer(), num_layers=2,
                 activation: Activation = A.self_normalizing_default(), transient: int = 30):
        super().__init__(
            reservoir=DeepESNCell(input_size, hidden_size, bias, initializer, num_layers, activation),
            readout=readout,
            transient=transient)


class GroupedDeepESN(ESNBase):
    def __init__(self, input_size: int = 1, hidden_size: int = 250, output_dim: int = 1, bias: bool = False,
                 initializer: WeightInitializer = WeightInitializer(), groups=2, num_layers=(2, 2),
                 activation: Activation =  A.self_normalizing_default(), transient: int = 30, regularization: float = 1.):
        super().__init__(
            reservoir=GroupOfESNCell(input_size, hidden_size, [
                DeepESNCell(input_size, hidden_size, bias, initializer, layers, activation) for layers in num_layers
            ], activation, bias, initializer),
            readout=SVDReadout(hidden_size * groups, output_dim, regularization=regularization),
            transient=transient)
