from matplotlib import pyplot as plt

import utils.dataset_loader as dl
from esn.esn import DeepESN
from esn.util import NRMSELoss

norm = False
mg17clean = dl.loader_explicit('datasets/mg.csv', 2_000, 6_000)
nrmse = NRMSELoss()


if norm:
    X, X_test, y, y_test, centr, spread = dl.norm_loader__(mg17clean)
    y_test = spread * y_test + centr
else:
    X, X_test, y, y_test = mg17clean()

esn = DeepESN()
esn.fit(X, y)

if norm:
    output = spread * esn(X_test) + centr
else:
    output = esn(X_test)

n = nrmse(output.unsqueeze(-1), y_test).item()
print(n)
last = 50
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()
