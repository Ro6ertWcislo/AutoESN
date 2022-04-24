import math
from enum import Enum

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn, Tensor
from typing import Union, List
from sklearn.metrics import accuracy_score
from .readout_mode import ReadoutMode

def create_dataloader(X, y, batch_size):
    data = []
    for i in range(len(X)):
        data.append([X[i], y[i]])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    return loader


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


class AutoNNReadout(nn.Module):
    def __init__(self, input_dim: int, reshape_factor, l2: float = None, l1: float = None, batch_size = 8, output_dim: int = 1, mode: ReadoutMode = ReadoutMode.Regression,
                 layers: Union[List[nn.Module], str] = 'auto', lr: float = 1e-4, epochs: int = 500, early_stop=None,
                 plateu=None, regul=None, device='cpu'):
        super().__init__()
        self.mode = mode
        self.model = torch.nn.Sequential(
            *(self._get_layers(input_dim, output_dim) if layers == "auto" else layers),
        )
        if self.mode == ReadoutMode.Regression:
            self.loss_fn = torch.nn.MSELoss()
        elif self.mode == ReadoutMode.BinaryClassification:
            self.loss_fn = torch.nn.BCELoss()
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)  # todo co to amsgrad?
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.99, nesterov=True)
        self.batch_size = batch_size
        self.epochs = epochs
        self.reshape_factor = reshape_factor
        self.l2 = l2
        self.l1 = l1
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []
        self.to_cuda()

    def _prepare_data_loaders(self, data_train, data_val):
        self.train_loader = create_dataloader(data_train[0], data_train[1], self.batch_size)
        self.val_loader = create_dataloader(data_val[0], data_val[1], self.batch_size)

    def _prepare_layers(self, input_dim: int, output_dim: int, mode: ReadoutMode):
        layers = []
        while input_dim > max(output_dim ** 2, 16):
            layers.append(nn.Linear(input_dim, int(math.sqrt(input_dim))))
            layers.append(nn.ReLU())
            input_dim = int(math.sqrt(input_dim))
        if mode == ReadoutMode.Regression:
            layers.append(nn.Linear(input_dim, output_dim))
        elif mode == ReadoutMode.BinaryClassification:
            layers.append(nn.Linear(input_dim, 2))
            layers.append(nn.Sigmoid())
        elif mode == ReadoutMode.MultiValueClassification:
            layers.append(nn.Linear(input_dim, output_dim))
            layers.append(nn.LogSoftmax())
        return layers

    def _get_layers(self, input_dim, output_dim):
        # DEFINE HERE YOUR MODEL
        layer_1 = nn.Linear(input_dim, 256)
        layer_2 = nn.Linear(256, 128)
        layer_out = nn.Linear(128, output_dim)
        relu = nn.ReLU()
        dropout = nn.Dropout(p=0.2)
        batch_norm1 = nn.BatchNorm1d(256)
        batch_norm3 = nn.BatchNorm1d(128)
        layers = [layer_1, batch_norm1, relu, dropout,   layer_2, relu, batch_norm3, relu, dropout,
                  layer_out, nn.Softmax(dim=2)]
        return layers

    def _compute_l2_loss(self, w):
        return torch.square(w).sum()

    def _compute_l1_loss(self, w):
        return torch.abs(w).sum()

    def _compute_loss(self, prediction, target):

        if self.mode == ReadoutMode.BinaryClassification:
            # FOR BINARY CROSS ENTROPY (BINARY)
            loss = self.loss_fn(prediction, target.to(torch.float).reshape(-1, 1))
            preds = prediction.reshape(-1).detach().cpu().numpy().round()
        else:
            # FOR CROSS ENTROPY (MULTICLASS)
            loss = self.loss_fn(prediction, target.to(torch.long))
            preds = np.argmax(prediction.detach().cpu().numpy().round(), axis=1)

        if self.l2 or self.l1:
            parameters = [p.view(-1) for p in self.model.parameters()]
            if self.l2:
                l2 = self.l2 * self._compute_l2_loss(torch.cat(parameters))
                loss += l2

            if self.l1:
                l1 = self.l1 * self._compute_l1_loss(torch.cat(parameters))
                loss += l1

        return loss, preds



    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    def fit(self):

        for t in range(self.epochs):
            curr_train_losses = []
            curr_val_losses = []
            curr_train_accs = []
            curr_val_accs = []

            # Forward pass: Compute predicted y by passing x to the model
            self.model.train()
            epoch = 0
            for i, (data, target) in enumerate(self.train_loader):
                if data.shape[0] == self.batch_size:
                    data, target = data.cuda(), target.cuda()
                    data = data.reshape(self.batch_size, self.reshape_factor)
                    prediction = self.model(data.reshape(self.batch_size, self.reshape_factor))
                    loss, predictions = self._compute_loss(prediction, target)
                    targets = target.detach().cpu().numpy()
                    acc = accuracy_score(targets, predictions)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    curr_train_losses.append(loss.item())
                    curr_train_accs.append(acc)
                epoch = i
            loss = torch.mean(torch.tensor(curr_train_losses))
            acc = torch.mean(torch.tensor(curr_train_accs))

            self.train_losses.append(loss)
            self.train_acc.append(acc)
            print(f"\t Epoch: {epoch} Train loss: {loss:.4f}")
            print(f"\t Epoch: {epoch} Train acc: {acc:.4f}")

            self.model.eval()
            with torch.no_grad():
                for data, target in self.val_loader:
                    if data.shape[0] == self.batch_size:
                        data, target = data.cuda(), target.cuda()
                        prediction = self.model(data.reshape(self.batch_size, self.reshape_factor))
                        loss, predictions = self._compute_loss(prediction, target)
                        targets = target.detach().cpu().numpy()
                        acc = accuracy_score(targets, predictions)
                        curr_val_losses.append(loss)
                        curr_val_accs.append(acc)

            loss = torch.mean(torch.tensor(curr_val_losses))
            acc = torch.mean(torch.tensor(curr_val_accs))

            self.val_losses.append(loss)
            self.val_acc.append(acc)
            print(f"\t Val loss: {loss:.5f}")
            print(f"\t Val acc: {acc:.4f}")

    def to_cuda(self):
        self.model.to('cuda')
