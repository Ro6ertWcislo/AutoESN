from matplotlib import pyplot as plt

import utils.dataset_loader as dl
from esn.esn import GroupedDeepESN
from esn.reservoir.util import NRMSELoss
"""
The easiest way to use the lib for typical 1-dim time series prediction
"""

# if you have a time series in .csv format, just make sure the values are under y column and use builtin loader
# In this example you'll end up with 600 train and 400 test samples.
# Leave max_samples empty and you'll get the whole series
mg17clean = dl.loader_explicit('datasets/mg10.csv', test_size=400, max_samples=1000)

# initialize loss function for evaluation
nrmse = NRMSELoss()

# run the loader
# If you don't want to use the loader just provide pytorch tensors of shape (num_samples,1,1)
X, X_test, y, y_test = mg17clean()

# initialize default ESN with 2 groups, 2 layers each, 250 reservoir in each layer and SNA activation
esn = GroupedDeepESN()

# fit
esn.fit(X, y)

# predict
output = esn(X_test)

# evaluate
n = nrmse(output.unsqueeze(-1), y_test).item()

print(n)

# visualize
last = 200
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()
