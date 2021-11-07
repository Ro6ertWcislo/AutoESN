from typing import Callable, Tuple, List, Union

import numpy as np
import pandas as pd
import torch.nn

device = torch.device('cpu')
dtype = torch.double
torch.set_default_dtype(dtype)


# todo ugly - to refactor

def load_train_testMVnext(pathOrDf: Union[str, pd.DataFrame], test_size: float, past: List[int]) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = (pd.read_csv(pathOrDf)['y'] if isinstance(pathOrDf,str) else pathOrDf).values
    data = np.array(data, dtype=np.float64)
    shift = max(past)

    X = torch.cat(
        [
            torch.from_numpy(data[shift - i:-1 - i].reshape((-1, 1, 1))).to(device)
            for i in past]
        , axis=2
    )
    y = data[shift + 1:].reshape((-1,  1))
    y = torch.from_numpy(y).to(device)

    return X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]


def load_train_test(pathOrDf: Union[str, pd.DataFrame], division: float, max_samples: int = -1) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = (pd.read_csv(pathOrDf)['y'] if isinstance(pathOrDf,str) else pathOrDf).values
    data = np.array(data, dtype=np.float64)
    X = data[:-1].reshape((-1, 1))[:max_samples]
    X = torch.from_numpy(X).to(device)
    y = data[1:].reshape((-1, 1))[:max_samples]
    y = torch.from_numpy(y).to(device)
    size = X.shape[0]
    p = int(size * division)
    return X[:p], X[p:], y[:p], y[p:]


def load_train_test2(pathOrDf: Union[str, pd.DataFrame], p: int, max_samples: int = -1) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = (pd.read_csv(pathOrDf)['y'] if isinstance(pathOrDf,str) else pathOrDf).values
    data = np.array(data, dtype=np.float64)

    X = data[:-1].reshape((-1, 1))[:max_samples]
    X = torch.from_numpy(X).to(device)
    y = data[1:].reshape((-1, 1))[:max_samples]
    y = torch.from_numpy(y).to(device)

    return X[:-p], X[-p:], y[:-p], y[-p:]


def load_train_test_val_test(pathOrDf: Union[str, pd.DataFrame], val_size=0.1, test_size=0.1, max_samples: int = -1) -> \
Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = (pd.read_csv(pathOrDf)['y'] if isinstance(pathOrDf,str) else pathOrDf).values
    data = np.array(data, dtype=np.float64)

    X = data[:-1].reshape((-1, 1))[:max_samples]
    X = torch.from_numpy(X).to(device)
    y = data[1:].reshape((-1, 1))[:max_samples]
    y = torch.from_numpy(y).to(device)

    if type(val_size) == float:
        val_size = int(val_size * X.size(0))
    if type(test_size) == float:
        test_size = int(test_size * X.size(0))

    return X[:-(val_size + test_size)], X[-(val_size + test_size):-test_size], X[-test_size:], y[:-(
            val_size + test_size)], y[-(val_size + test_size):-test_size], y[-test_size:]


def load_train_test_memory_capacity(pathOrDf: Union[str, pd.DataFrame], test_size: int, past: int) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = (pd.read_csv(pathOrDf)['y'] if isinstance(pathOrDf,str) else pathOrDf).values
    data = np.array(data, dtype=np.float64)

    X = data[past:].reshape((-1, 1))
    X = torch.from_numpy(X).to(device)
    y = data[:-past].reshape((-1, 1))
    y = torch.from_numpy(y).to(device)

    return X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]


def loader_memory_capacity(pathOrDf: Union[str, pd.DataFrame], test_size: int, past: int) -> Callable:
    def loader_():
        return load_train_test_memory_capacity(pathOrDf, test_size, past)

    return loader_


def loader(pathOrDf: Union[str, pd.DataFrame], division: float, max_samples: int = -1) -> Callable:
    def loader_():
        return load_train_test(pathOrDf, division, max_samples)

    return loader_


def loader_MV_with_past(pathOrDf: Union[str, pd.DataFrame], test_size: int, past: List[int]):
    def loader_():
        return load_train_testMVnext(pathOrDf, test_size, past)

    return loader_


def loader_explicit(pathOrDf: Union[str, pd.DataFrame], test_size: int, max_samples: int = -1) -> Callable:
    def loader_():
        return load_train_test2(pathOrDf, test_size, max_samples)

    return loader_


def loader_val_test(pathOrDf: Union[str, pd.DataFrame], val_size=0.1, test_size=0.1, max_samples: int = -1) -> Callable:
    def loader_():
        return load_train_test_val_test(pathOrDf, val_size, test_size, max_samples)

    return loader_


def load_train_test_ahead(pathOrDf: Union[str, pd.DataFrame], p: int, ahead: int = 1, max_samples: int = -1) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = (pd.read_csv(pathOrDf)['y'] if isinstance(pathOrDf,str) else pathOrDf).values
    data = np.array(data, dtype=np.float64)

    X = data[:-ahead].reshape((-1, 1))[:max_samples]
    X = torch.from_numpy(X).to(device)
    y = data[ahead:].reshape((-1, 1))[:max_samples]
    y = torch.from_numpy(y).to(device)

    return X[:-p], X[-p:], y[:-p], y[-p:]


def loader_explicit_ahead(pathOrDf: Union[str, pd.DataFrame], test_size: int, ahead: int = 1, max_samples: int = -1):
    def loader_():
        return load_train_test_ahead(pathOrDf, test_size, ahead, max_samples)

    return loader_


def norm_loader__(loader):
    X, X_test, y, y_test = loader()

    def coeff(X, X_test):
        x = torch.cat([X, X_test]).squeeze().squeeze()
        return torch.mean(x).item(), torch.max(x).item() - torch.min(x).item()

    centr, spread = coeff(X, X_test)

    def norm(X, centr, spread):
        return (X - centr) / spread

    def n(X):
        return norm(X, centr, spread)

    return n(X), n(X_test), n(y), n(y_test), centr, spread


def norm_loader_val_test_(loader):
    X, X_val, X_test, y, y_val, y_test = loader()

    def coeff(X, X_val, X_test):
        x = torch.cat([X, X_val, X_test]).squeeze().squeeze()
        return torch.mean(x).item(), torch.max(x).item() - torch.min(x).item()

    centr, spread = coeff(X, X_val, X_test)

    def norm(X, centr, spread):
        return (X - centr) / spread

    def n(X):
        return norm(X, centr, spread)

    return n(X), n(X_val), n(X_test), n(y), n(y_val), n(y_test), centr, spread
