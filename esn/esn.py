import torch
from torch import nn
from esn import activation as A
from esn.initialization import WeightInitializer


class ESNCellBase(nn.Module):
    r"""
    Slightly adjusted version of torch.nn.RNNCellBase
    Main changes includes flexible initialization.
    """
    __constants__ = ['input_size', 'hidden_size', 'bias']

    def __init__(self, input_size, hidden_size, bias, initializer: WeightInitializer = None, num_chunks=1):
        super(ESNCellBase, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size, hidden_size))
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.initializer = initializer
        self.reset_parameters()

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        # type: (Tensor, Tensor, str) -> None
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        self.initializer.init_weight_ih(self.weight_ih, self.hidden_size)
        self.initializer.init_weight_hh(self.weight_hh, self.hidden_size)
        if self.bias:
            self.initializer.init_bias_ih(self.bias_ih, self.hidden_size)
            self.initializer.init_bias_ih(self.bias_hh, self.hidden_size)


class ESNCell(ESNCellBase):
    # todo type annotations
    __constants__ = ['input_size', 'hidden_size', 'bias', 'activation']

    def __init__(self, input_size, hidden_size, bias=True, initializer: WeightInitializer = None, activation=A.tanh):
        super(ESNCell, self).__init__(input_size, hidden_size, bias, initializer=initializer, num_chunks=1)
        self.activation = activation
        self.hx = None

    def forward(self, input):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        self.check_forward_input(input)
        if self.hx is None:
            self.hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device,
                                  requires_grad=False)
        self.check_forward_hidden(input, self.hx, '')
        self.hx = self.activation(
            input, self.hx,
            self.weight_ih, self.weight_hh,
            self.bias_ih, self.bias_hh,
        )
        return self.hx


class DeepESNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, initializer: WeightInitializer = None, num_layers=1,
                 activation=A.tanh):
        super().__init__()
        self.activation = activation
        self.layers = [ESNCell(input_size, hidden_size, bias, initializer, activation)] + \
                      [ESNCell(hidden_size, hidden_size, bias, initializer, activation) for i in
                       range(1, num_layers)]
        # self.hidden_states = [
        #     torch.zeros(input_size, hidden_size, requires_grad=False)]  # todo check if good dimension for input
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.initializer = initializer  # todo generalize?
        self.activation = activation  # todo generalize?

    def forward(self, input):
        # todo obsługa transient
        result = torch.Tensor(input.size(0), len(self.layers) * self.hidden_size)

        # todo 1.  transient dodać, 2. debug czy dalej nie działa, sprawdzic inicjalizacje, ogarnac washout i prepare costam
        for i in range(input.size(0)):
            cell_input = input[i]
            new_hidden_states = []
            for esn_cell in self.layers:
                cell_input = esn_cell(cell_input)
                new_hidden_states.append(cell_input)
            # self.hidden_states = new_hidden_states  # todo potrzebne w ogole
            result[i, :] = torch.cat(new_hidden_states, axis=1)  # moze to tu?
        return result

    def washout(self, input):
        for i in range(input.size(0)):
            cell_input = input[i]
            for esn_cell in self.layers:
                cell_input = esn_cell(cell_input)


class SVDReadout(nn.Module):
    def __init__(self, total_hidden_size, output_dim, regularization=1):
        super().__init__()
        self.readout = nn.Linear(total_hidden_size, output_dim)
        self.regularization = regularization

    def forward(self, input):
        return self.readout(input)

    def fit(self, input, target):
        X = torch.ones(input.size(0), 1 + input.size(1), device=target.device)
        X[:, :-1] = input
        W = self._solve_svd(X, target, self.regularization)
        self.readout.bias = nn.Parameter(W[:, -1], requires_grad=False)
        self.readout.weight = nn.Parameter(W[:, :-1], requires_grad=False)

    def _solve_svd(self, X, y, alpha):
        y = y[:, 0, :]  # ignore batch # todo stack by batch
        U, s, V = torch.svd(X)
        idx = s > 1e-15  # same default value as scipy.linalg.pinv
        s_nnz = s[idx][:, None]
        UTy = U.T @ y  # UTy = torch.mm(U.t(), target)
        d = torch.zeros(s.size(0), 1, device=X.device)
        d[idx] = s_nnz / (s_nnz ** 2 + alpha)
        d_UT_y = d * UTy

        return (V @ d_UT_y).T  # trchmm?


class DeepESN(nn.Module):
    def __init__(self, input_size, hidden_size, output_dim=1, bias=True, initialization=None, num_layers=1,
                 activation=A.tanh, transient=30, reglarization=1):
        super(DeepESN, self).__init__()
        self.transient = transient
        self.initial_state = True
        self.reservoir = DeepESNCell(input_size, hidden_size, bias, initialization, num_layers, activation)
        self.readout = SVDReadout(hidden_size * num_layers, output_dim, regularization=reglarization)

    def fit(self, input, target):
        if self.initial_state:
            self.initial_state = False
            self.reservoir.washout(input[:self.transient])
            mapped_input = self.reservoir(input[self.transient:])
            self.readout.fit(mapped_input, target[self.transient:])
        else:
            mapped_input = self.reservoir(input)
            self.readout.fit(mapped_input, target)

    def forward(self, input):
        self.initial_state = False
        mapped_input = self.reservoir(input)

        return self.readout(mapped_input)

# todo GPU, proper nograd, typing
