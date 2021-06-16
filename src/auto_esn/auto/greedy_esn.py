import logging
import random
import time
from typing import Tuple, List

from torch import nn, Tensor

from auto_esn.auto.util import next_gen, random_gen
from auto_esn.esn.reservoir import activation
from auto_esn.esn.esn import DeepESN
from auto_esn.esn.reservoir.util import NRMSELoss
from auto_esn.utils.types import IntGen, FloatGen

default_size_gen = lambda: random.choice([20, 30, ])


# todo add normalization


class GreedyESN(nn.Module):
    def __init__(self,
                 max_samples: int = 20,
                 max_time_sec: int = 600,
                 size_gen: IntGen = next_gen([100, 250, 500, 1000]),
                 layer_gen: IntGen = random_gen([1, 1, 1, 2, 2, 3, 4]),
                 leaky_gen: FloatGen = random_gen([0.4,0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
                 metric=NRMSELoss(),
                 fast=True,
                 nbest=0
                 ):
        super().__init__()
        self.max_samples = max_samples
        self.max_time_sec = max_time_sec
        self.size_gen = size_gen
        self.layer_gen = layer_gen
        self.leaky_gen = leaky_gen
        self.metric = metric
        self.fast = fast
        self.models = []
        self.nbest=nbest

    def fit(self, X: Tensor, y: Tensor, X_val: Tensor, y_val: Tensor):
        start = time.time()

        sample_no = 0
        results: List[Tuple[float, Tensor]] = []
        models = []
        while sample_no < self.max_samples and time.time() - start < self.max_time_sec:
            size, layers, leaky = self.size_gen(), self.layer_gen(), self.leaky_gen()
            logging.info(f"sample no.{sample_no} with layers ={layers}, size={size}, leaky={leaky}")
            esn = DeepESN(
                num_layers=layers,
                hidden_size=size,
                activation = activation.self_normalizing_default(spectral_radius=100.0, leaky_rate=leaky),
                # readout = AutoNNReadout(input_dim=layers*size, lr=1e-4, epochs=1700)
            )
            esn.fit(X, y)
            output = esn(X_val)

            act_metric = self.metric(output.unsqueeze(-1), y_val).item()
            logging.info(f"sample no.{sample_no} trained with {self.metric.__name__} = {act_metric} ")
            results.append((act_metric, output.unsqueeze(-1)))
            models.append(esn)

            sample_no+=1

        results = sorted(results, key=lambda x: x[0])  # todo handle norm data
        if self.nbest > 0:
            used = set(range(self.nbest))
            self.models = [models[i] for i in used]
            curr_out = sum([results[i][1] for i in used]) / len(used)
            logging.info(f"grouping improved {results[0][0]} to {self.metric(curr_out,y_val)} by merging models: {used}")
            return
        if self.fast:
            best_metric = results[0][0]
            used = {0}
        else:

            grid = [
                (i, j, self.metric((results[i][1] + results[j][1]) / 2, y_val))
                for i in range(len(results) - 1)
                for j in range(i, len(results))
            ]
            min_grid = min(grid, key=lambda x: x[2])
            best_metric = min_grid[2]
            used = {min_grid[0], min_grid[1]}
        clean_pass = False
        while not clean_pass:
            curr_out = sum([results[i][1] for i in used])
            new_groups = [(i, self.metric((results[i][1] + curr_out) / (len(used) + 1), y_val).item())
                          for i
                          in set(range(len(results))).difference(used)]
            best_idx, best_curr = max(new_groups, key=lambda x: x[1])
            if best_curr < best_metric:
                used.add(best_idx)
                best_metric = best_curr
            else:
                clean_pass = True

        self.models = [models[i] for i in used]
        logging.info(f"grouping improved {results[0][0]} to {best_metric} by merging models: {used}")

    def forward(self, input: Tensor) -> Tensor:
        return sum([model(input) for model in self.models]) / len(self.models) # todo make it torch?
