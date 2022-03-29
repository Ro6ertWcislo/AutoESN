from matplotlib import pyplot as plt

from auto_esn.datasets.predefined import DatasetType, PredefinedDataset
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from auto_esn.esn.reservoir.util import NRMSELoss

# You can also use PredefinedDataset if you wish
dataset = PredefinedDataset(DatasetType.SUNSPOT).load(val_size=300, test_size=300)

nrmse = NRMSELoss()


# example complex usage of initialization method
def regular_graph_initializer(seed, degree):
    # initialize input weights with uniform distribution from -1 to 1 and specified seed to reproduce results
    input_weight = CompositeInitializer().with_seed(seed).uniform()

    # specified operations will be done one by one, so this "builder" can be seen as a list of transforms
    # first set the seed and start with uniform distribution
    # then treat the newly created dense matrix as adjacency matrix and transform it into regular graph with
    # desired degree, then apply spectral normalization, so that spectral radius is 1.
    # at the end scale the matrix by factor 0.9 and the initialization is done
    reservoir_weight = CompositeInitializer() \
        .with_seed(seed) \
        .uniform() \
        .regular_graph(degree) \
        .spectral_normalize() \
        .scale(0.9)

    return WeightInitializer(weight_ih_init=input_weight, weight_hh_init=reservoir_weight)


# now choose activation and configure it
activation = self_normalizing_default(leaky_rate=1.0, spectral_radius=500)

# initialize the esn
esn = GroupedDeepESN(
    groups=4,  # choose number of groups
    num_layers=(1, 2, 3, 4),  # choose number of layers for each group
    hidden_size=80,  # choose hidden size for all reservoirs
    initializer=regular_graph_initializer(seed=3, degree=6),  # choose 6-regular graph as reservoir structure
    activation=activation  # assign activation
)

# fit
esn.fit(dataset.x_train, dataset.y_train)

# predict
val_output = esn(dataset.x_val)
test_output = esn(dataset.x_test)

# evaluate
err_val = nrmse(val_output, dataset.y_val).item()
print(f"Validation error: {err_val}")

err_test = nrmse(test_output, dataset.y_test).item()
print(f"Test error: {err_test}")

# plot validation set
last = 200
plt.plot(range(last), val_output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), dataset.y_val.view(-1).detach().numpy()[-last:], 'b')
plt.title("Validation set results")
plt.show()

# plot test test
plt.plot(range(last), test_output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), dataset.y_test.view(-1).detach().numpy()[-last:], 'b')
plt.title("Test set results")
plt.show()
