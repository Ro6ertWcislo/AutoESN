import math
from enum import Enum

import torch
from torch import nn, Tensor
from typing import Union, List


class ReadoutMode(Enum):
    Regression = 1
    BinaryClassification = 2
    MultiValueClassification = 3

class AutoNNReadout(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1, mode: ReadoutMode = ReadoutMode.Regression,
                 layers: Union[List[nn.Module], str] = 'auto', lr: float = 1e-4, epochs: int = 500, early_stop=None,
                 plateu=None, regul=None, device='cpu'):
        super().__init__()
        self.model = torch.nn.Sequential(
            *(self._prepare_layers(input_dim, output_dim, mode) if layers == "auto" else layers),
        )
        if mode == ReadoutMode.Regression:
            self.loss_fn = torch.nn.MSELoss()
        elif mode == ReadoutMode.BinaryClassification:
            self.loss_fn = torch.nn.BCELoss()
        else:
            self.loss_fn = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) # todo co to amsgrad?
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.99, nesterov=True)

        self.epochs = epochs

    def _prepare_layers(self, input_dim: int, output_dim: int, mode: ReadoutMode):
        layers = []
        while input_dim > max(output_dim**2, 16):
            layers.append(nn.Linear(input_dim, int(math.sqrt(input_dim))))
            layers.append(nn.ReLU())
            input_dim = int(math.sqrt(input_dim))
        if mode == ReadoutMode.Regression:
            layers.append(nn.Linear(input_dim, output_dim))
        elif mode == ReadoutMode.BinaryClassification:
            layers.append(nn.Linear(input_dim, 1))
            layers.append(nn.Sigmoid())
        elif mode == ReadoutMode.MultiValueClassification:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LogSoftmax())
        return layers

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    def fit(self, X: Tensor, y: Tensor):
        if X.ndim == 2:
            X = X.unsqueeze(0)
        y = y.reshape(1, -1, 1)
        for t in range(self.epochs):
            # Forward pass: Compute predicted y by passing x to the model
            y_pred = self.model(X)

            # Compute and print loss
            loss = self.loss_fn(y_pred, y)
            if t % 50 == 49:
                print(t, loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def to_cuda(self):
        self.model.to('cuda')
