import torch
from matplotlib import pyplot as plt

from auto_esn.datasets.predefined import DatasetType, PredefinedDataset
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from auto_esn.esn.reservoir.util import NRMSELoss

extrapolation_steps = 100

# You can also use PredefinedDataset if you wish
dataset = PredefinedDataset(DatasetType.MackeyGlass).load(val_size=0, test_size=extrapolation_steps)

nrmse = NRMSELoss()


# example complex usage of initialization method
def regular_graph_initializer(seed, degree):
    # initialize input weights with uniform distribution from -1 to 1 and specified seed to reproduce results
    input_weight = CompositeInitializer().with_seed(seed).uniform()

    reservoir_weight = CompositeInitializer() \
        .with_seed(seed) \
        .uniform() \
        .regular_graph(degree) \
        .spectral_normalize() \
        .scale(1.)

    return WeightInitializer(weight_ih_init=input_weight, weight_hh_init=reservoir_weight)


# now choose activation and configure it
activation = self_normalizing_default(leaky_rate=1., spectral_radius=500)

# initialize the esn
esn = GroupedDeepESN(
    groups=3,  # choose number of groups
    num_layers=(3, 3, 3),  # choose number of layers for each group
    hidden_size=250,  # choose hidden size for all reservoirs
    initializer=regular_graph_initializer(seed=3, degree=50),  # choose 50-regular graph as reservoir structure
    regularization=0.1,
    activation=activation  # assign activation
)

# fit
esn.fit(dataset.x_train, dataset.y_train)

# esn already has the state after consuming whole training dataset
# let's start from first element in test dataset and let it extrapolate further
val = dataset.x_test[0:1]
result = []
for j in range(extrapolation_steps):  # 100 steps ahead
    val = esn(val)  # (1,1) tensor
    result.append(val)

res = torch.vstack(result)

# evaluate
err = nrmse(res, dataset.y_test).item()
print(f"Extrapolation error: {err}")

# plot validation set

plt.plot(range(extrapolation_steps), res.view(-1).detach().numpy(), 'r', )
plt.plot(range(extrapolation_steps), dataset.y_test.view(-1).detach().numpy(), 'b')
plt.title("Validation set results")
plt.show()
