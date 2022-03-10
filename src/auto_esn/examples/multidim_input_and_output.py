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

print(f"Size of X: {X.shape}, X_test: {X_test.shape}, y: {y.shape}, y_test: {y_test.shape}")
# double the dimensionality of test and train input
# second series is shifted by 1
X = torch.cat((X, X - 1), dim=1)
y = torch.cat((y, y - 1), dim=1)
# double the dimensionality of test and train output
# second series is shifted by 1
X_test = torch.cat((X_test, X_test - 1), dim=1)
y_test = torch.cat((y_test, y_test - 1), dim=1)
print(f"Size of doubled X: {X.shape}, X_test: {X_test.shape}, y: {y.shape}, y_test: {y_test.shape}")

# now choose activation and configure it
activation = self_normalizing_default(leaky_rate=1.0, spectral_radius=500)

# initialize the esn
esn = GroupedDeepESN(
    # You need to specify the dimensionality of the input
    # it will later be checked whether the data provided matches the declared shape
    input_size=2,
    groups=4,
    num_layers=(2, 2, 2, 2),
    hidden_size=80,
    activation=activation,
    # You need to specify the dimensionality of the output
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
# we have 2dimensional input, so we have to plot two series for ground truth and for prediction
# plot original prediction
plt.plot(range(last), output[:, 0].view(-1).detach().numpy()[-last:], 'r')
# plot shifted prediction
plt.plot(range(last), output[:, 1].view(-1).detach().numpy()[-last:], 'r')
# plot original ground truth
plt.plot(range(last), y_test[:, 0].view(-1).detach().numpy()[-last:], 'b')
# plot shifted ground truth
plt.plot(range(last), y_test[:, 1].view(-1).detach().numpy()[-last:], 'b')
plt.show()
