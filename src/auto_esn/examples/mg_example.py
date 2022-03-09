import torch
from matplotlib import pyplot as plt

import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import MackeyGlass
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from auto_esn.esn.reservoir.util import NRMSELoss


mg17clean = dl.loader_explicit(MackeyGlass, test_size=400)
nrmse = NRMSELoss()

X, X_test, y, y_test = mg17clean()

print(f"Size of X: {X.shape}, X_test: {X_test.shape}")
X = torch.cat((X,X-1),dim=1)   
y = torch.cat((y,y-1),dim=1)   
X_test = torch.cat((X_test,X_test-1),dim=1)
y_test = torch.cat((y_test,y_test-1),dim=1)
print(f"Size of doubled X: {X.shape}, X_test: {X_test.shape}")

# example complex usage of initialization method
def regular_graph_initializer(seed, degree):
    # initialize input weights with uniform distribution from -1 to 1 and specified seed to reproduce results
    input_weight =CompositeInitializer().with_seed(seed).uniform()

    # specified operations will be done one by one, so this "builder" can be seen as a list of transforms
    # first set the seed and start with uniform distribution
    # then treat the newly created dense matrix as adjacency matrix and transform it into regular graph with
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
    input_size=2, 
    groups=4,                   # choose number of groups
    num_layers=(1, 2, 3, 4),    # choose number of layers for each group
    hidden_size=80,             # choose hidden size for all reservoirs
    initializer=regular_graph_initializer(seed=3, degree=6),  # choose 6-regular graph as reservoir structure
    activation=activation,       # assign activation
    output_dim=2 
)

# fit
esn.fit(X, y)

# predict
output = esn(X_test)
print(f"Shape of output: {output.shape}")


# evaluate
n = nrmse(output, y_test).item()
print(n)

# plot
last = 200
plt.plot(range(last), output[:,0].view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), output[:,1].view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test[:,0].view(-1).detach().numpy()[-last:], 'b')
plt.plot(range(last), y_test[:,1].view(-1).detach().numpy()[-last:], 'b')
plt.show()
