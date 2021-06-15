from matplotlib import pyplot as plt

import utils.dataset_loader as dl
from esn.esn import GroupedDeepESN
from esn.reservoir.activation import self_normalizing_default
from esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from esn.reservoir.util import NRMSELoss


mg17clean = dl.loader_explicit('datasets/mg10.csv', test_size=400)
nrmse = NRMSELoss()

X, X_test, y, y_test = mg17clean()

# example complex usage of initialization method
def regular_graph_initializer(seed, degree):
    # initialize input weights with uniform distribution from -1 to 1 and specified seed to reproduce results
    input_weight =CompositeInitializer().with_seed(seed).uniform()

    # specified operations will be done one by one, can be seen as a list of transforms
    # first set the seed and start with uniform distribution
    # then treat the newly created dense matrix as adjacency matrix and transform is into regular graph with
    # desired degree, then apply spectral normalization, so that spectral radius is 1.
    # at the end scale the matrix by factor 0.9 and the initialization is done
    reservoir_weight = CompositeInitializer()\
      .with_seed(seed) \
      .uniform()\
      .regular_graph(degree)\
      .spectral_normalize()\
      .scale(0.9)

    return WeightInitializer(weight_ih_init=input_weight, weight_hh_init=reservoir_weight)


# now choose activation and configure it
activation = self_normalizing_default(leaky_rate=1.0, spectral_radius=500)

# initialize the esn
esn = GroupedDeepESN(
    groups=4,                   # choose number of groups
    num_layers=(1, 2, 3, 4),    # choose number of layers for each group
    hidden_size=80,             # choose hidden size for all reservoirs
    initializer=regular_graph_initializer(seed=3, degree=6),  # choose 6-regular graph as reservoir structure
    activation=activation       # assign activation
)

# fit
esn.fit(X, y)

# predict
output = esn(X_test)

# evaluate
n = nrmse(output.unsqueeze(-1), y_test).item()
print(n)

# plot
last = 200
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()
