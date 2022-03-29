import numpy as np
from matplotlib import pyplot as plt

import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import MackeyGlass
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from auto_esn.esn.reservoir.util import NRMSELoss

"""
The easiest way to use the lib for typical 1-dim time series prediction
"""

mg = dl.loader_val_test(MackeyGlass, val_size=200, test_size=200)

x_train, x_val, x_test, y_train, y_val, y_test, baseline, spread = dl.norm_loader_val_test_(mg)
y_val = spread * y_val + baseline
y_test = spread * y_test + baseline

# initialize loss function for evaluation
nrmse = NRMSELoss()


def best_deepesn_initializer(seed):
    # initialize input weights with uniform distribution from -1 to 1 and specified seed to reproduce results
    input_weight = CompositeInitializer().with_seed(seed).uniform()

    reservoir_weight = CompositeInitializer() \
        .with_seed(seed) \
        .uniform() \
        .sparse(density=0.1) \
        .spectral_normalize() \
        .scale(factor=1.0)  # unnecesary but i wanted to make it explicit

    return WeightInitializer(weight_ih_init=input_weight, weight_hh_init=reservoir_weight)


# initialize default ESN with 2 groups, 2 layers each, 250 reservoir in each layer and SNA activation
activation = self_normalizing_default(leaky_rate=1.0, spectral_radius=100)

# initialize the esn
best_output = None
min_test_score = float("inf")
test_scores = []
for seed in [i for i in range(6, 128, 7)]:
    esn = GroupedDeepESN(
        groups=1,
        num_layers=(3,),
        hidden_size=334,
        activation=activation,
        initializer=best_deepesn_initializer(seed),
        regularization=0.5,
        washout=100
    )

    # fit
    esn.fit(x_train, y_train)

    # predict on validation set
    output_val = spread * esn(x_val) + baseline
    output_test = spread * esn(x_test) + baseline

    # evaluate
    score_val = nrmse(output_val, y_val).item()
    score_test = nrmse(output_test, y_test).item()
    test_scores.append(score_test)
    if score_test < min_test_score:
        min_test_score = score_test
        best_output = output_test

    print(f"Validation NRMSE: {score_val}, test NRMSE: {score_test} for seed = {seed}")

print('\n ####################')
print(f'Best score obtained on test set was: {min_test_score}')
print(test_scores)
print(f'Average score on test set was: {np.mean(test_scores)}')
print('####################\n')
plt.plot(range(200), best_output.view(-1).detach().numpy(), 'r')
plt.plot(range(200), y_test.view(-1).detach().numpy(), 'b')
plt.show()
