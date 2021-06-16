from matplotlib import pyplot as plt

import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import SunspotSmooth
from auto_esn.esn.readout.nn_readout import AutoNNReadout
from auto_esn.esn.reservoir import activation
from auto_esn.esn.esn import FlexDeepESN

from auto_esn.esn.reservoir.util import NRMSELoss

norm = True
sunspotSmooth = dl.loader_explicit(SunspotSmooth, test_size=600)
nrmse = NRMSELoss()
if norm:
    X, X_test, y, y_test, centr, spread = dl.norm_loader__(sunspotSmooth)
    y_test = spread * y_test + centr
else:
    X, X_test, y, y_test = sunspotSmooth()

# this can be used to plug a neural network as readout
esn = FlexDeepESN(
    hidden_size=600,
    num_layers=2,
    activation=activation.self_normalizing_default(spectral_radius=100.0),
    readout=AutoNNReadout(input_dim=1200, lr=1e-4, epochs=1000)
)

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
