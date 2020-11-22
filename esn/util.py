import torch
from torch import Tensor


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
