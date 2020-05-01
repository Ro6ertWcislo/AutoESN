from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch.nn

device = torch.device('cpu')
dtype = torch.double
torch.set_default_dtype(dtype)


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
