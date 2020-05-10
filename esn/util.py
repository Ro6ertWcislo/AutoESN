from copy import deepcopy

from torch import Tensor

from esn.esn import DeepSubreservoirESN
from utils.types import Metric


def fit_transform_DSESN(esn: DeepSubreservoirESN, input: Tensor, target: Tensor, test_input: Tensor,
                        test_target: Tensor, metric: Metric, verbose=0):
    # increase subreservoir size until it gets better on test set
    # todo optimize- can be handled with more efficient copying!-pass reference to weight matrices instead o copying them
    best_model = esn
    esn.fit(input, target)
    output = esn(test_input)
    best_output = output
    best_result = float('inf')
    new_result = metric(output.unsqueeze(-1), test_target).item()

    while new_result < best_result:
        if verbose > 0:
            print(f'growing improved {getattr(metric,"__name__",type(metric).__name__)} from {best_result} to {new_result}')  # todo add logging
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
