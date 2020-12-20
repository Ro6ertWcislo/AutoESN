import random

import torch
from networkx import random_regular_graph, adjacency_matrix, is_connected
from torch import Tensor
import numpy as np


def NRMSELoss():
    def _NRMSELoss(yhat, y):
        avg = torch.mean(y)
        res = yhat - y
        res_avg = yhat - avg

        return torch.sqrt(torch.sum(torch.mul(res, res)) / torch.sum(torch.mul(res_avg, res_avg)))

    return _NRMSELoss


def R2_score():
    def _R2_score(yhat, y: Tensor):
        var = torch.sum((y - torch.mean(y)) ** 2)
        residuals_sq = torch.sum((y - yhat) ** 2)

        return 1 - residuals_sq.item() / var.item()

    return _R2_score


def NRMSELossMG():
    def _NRMSELossMG(yhat, y):
        res = yhat - y

        return torch.sqrt(torch.sum(res ** 2) / y.var())

    return _NRMSELossMG


def RMSELoss():
    def _RMSELoss(yhat, y):
        return torch.sqrt(torch.mean((yhat - y) ** 2))

    return _RMSELoss


def get_star_graph_mask(size: int, stars: int = 1) -> Tensor:
    available_range = list(range(size))
    rows_to_zero = random.sample(available_range, stars)
    cols_to_zero = random.sample(available_range, stars)

    multi_star_mask = torch.zeros(size, size)
    multi_star_mask[rows_to_zero, :] = 1
    multi_star_mask[:, cols_to_zero] = 1

    return multi_star_mask


def get_regular_graph_mask(degree: int, nodes: int, max_sample=50) -> Tensor:
    i = 0
    G = random_regular_graph(d=degree, n=nodes)
    while not is_connected(G):
        i += 1
        G = random_regular_graph(d=degree, n=nodes)
        if i > max_sample:
            raise RuntimeError("could not draw random internal graph mask for weight matrix")

    return torch.from_numpy(adjacency_matrix(G).toarray())


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
