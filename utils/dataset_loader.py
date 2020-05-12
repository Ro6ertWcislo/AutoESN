from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch.nn

device = torch.device('cpu')
dtype = torch.double
torch.set_default_dtype(dtype)


# todo ugly - to refactor

def load_train_testXY(path: str, test_size: float) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dataX = pd.read_csv(path)['x'].values
    dataY = pd.read_csv(path)['y'].values
    dataX = np.array(dataX, dtype=np.float64)
    dataY = np.array(dataY, dtype=np.float64)

    X = dataX.reshape((-1, 1, 1))
    X = torch.from_numpy(X).to(device)
    y = dataY.reshape((-1, 1, 1))
    y = torch.from_numpy(y).to(device)

    return X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]


def load_train_test(path: str, division: float, max_samples: int = -1) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = pd.read_csv(path)['y'].values
    data = np.array(data, dtype=np.float64)
    X = data[:-1].reshape((-1, 1, 1))[:max_samples]
    X = torch.from_numpy(X).to(device)
    y = data[1:].reshape((-1, 1, 1))[:max_samples]
    y = torch.from_numpy(y).to(device)
    size = X.shape[0]
    p = int(size * division)
    return X[:p], X[p:], y[:p], y[p:]


def load_train_test2(path: str, p: float, max_samples: int = -1) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = pd.read_csv(path)['y'].values
    data = np.array(data, dtype=np.float64)

    X = data[:-1].reshape((-1, 1, 1))[:max_samples]
    X = torch.from_numpy(X).to(device)
    y = data[1:].reshape((-1, 1, 1))[:max_samples]
    y = torch.from_numpy(y).to(device)

    return X[:-p], X[-p:], y[:-p], y[-p:]


def loader(path: str, division: float, max_samples: int = -1) -> Callable:
    def loader_():
        return load_train_test(path, division, max_samples)

    return loader_


def loader_explicit(path: str, division: float, max_samples: int = -1) -> Callable:
    def loader_():
        return load_train_test2(path, division, max_samples)

    return loader_


def loaderXY(path: str, test_size: int) -> Callable:
    def loader_():
        return load_train_testXY(path, test_size)

    return loader_
