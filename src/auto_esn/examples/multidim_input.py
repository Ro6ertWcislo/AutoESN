import torch
from matplotlib import pyplot as plt

import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import MackeyGlass
from auto_esn.esn.esn import GroupedDeepESN
from auto_esn.esn.reservoir.activation import self_normalizing_default
from auto_esn.esn.reservoir.util import NRMSELoss

mg17clean = dl.loader_explicit(MackeyGlass, test_size=400)
nrmse = NRMSELoss()

X, X_test, y, y_test = mg17clean()

print(f"Size of X: {X.shape}, X_test: {X_test.shape}")
# double the dimensionality of test and train input
X = torch.cat((X, X), dim=1)
X_test = torch.cat((X_test, X_test), dim=1)
print(f"Size of doubled X: {X.shape}, X_test: {X_test.shape}")

# now choose activation and configure it
activation = self_normalizing_default(leaky_rate=1.0, spectral_radius=500)

# initialize the esn
esn = GroupedDeepESN(
    # You need to specify the dimensionality of the input
    # it will later be checked whether the data provided matches the declared shape
    input_size=2,
    groups=3,
    num_layers=(2, 2, 3),
    hidden_size=80,
    activation=activation
)

# fit
esn.fit(X, y)
# predict
output = esn(X_test)
# evaluate
n = nrmse(output, y_test).item()
print(n)
# plot
last = 200
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()
