from matplotlib import pyplot as plt

import utils.dataset_loader as dl
from esn.esn import GroupedDeepESN
from esn.reservoir.util import NRMSELoss

"""
The easiest way to use the lib for typical 1-dim time series prediction
"""

mg17clean = dl.loader_explicit('datasets/mg10.csv', test_size=400)
nrmse = NRMSELoss()

X, X_test, y, y_test = mg17clean()

# esn = GroupOfESN(groups=4, hidden_size=250)
# esn = DeepESN()
esn = GroupedDeepESN(groups=4, num_layers=(1, 2, 3, 4), hidden_size=80)
esn.fit(X, y)
output = esn(X_test)

n = nrmse(output.unsqueeze(-1), y_test).item()
print(n)

last = 50
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()
