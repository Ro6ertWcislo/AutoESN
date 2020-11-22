from copy import deepcopy

from torch import Tensor
import torch

from esn.esn import DeepSubreservoirESN
from utils.types import Metric


def fit_transform_DSESN(esn: DeepSubreservoirESN, input: Tensor, target: Tensor, test_input: Tensor,
                        test_target: Tensor, metric: Metric, verbose=0, max_neurons=10_000):
    # increase subreservoir size until it gets better on test set
    # todo optimize- can be handled with more efficient copying!-pass reference to weight matrices instead o copying them
    best_model = esn
    esn.fit(input, target)
    output = esn(test_input)
    best_output = output
    best_result = float('inf')
    new_result = metric(output.unsqueeze(-1), test_target).item()

    while new_result < best_result and best_model.get_number_of_neurons()<max_neurons:
        if verbose > 0:
            print(
                f'growing improved {getattr(metric, "__name__", type(metric).__name__)} from {best_result} to {new_result}')  # todo add logging
        best_result = new_result
        best_model = deepcopy(esn)
        best_output = output
        esn.reset_hidden()
        esn.grow()
        esn.fit(input, target)
        output = esn(test_input)
        # single batch?
        new_result = metric(output.unsqueeze(-1), test_target).item()

    return best_model, best_output


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

    from sklearn.metrics import r2_score

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
