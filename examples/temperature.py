from matplotlib import pyplot as plt

import utils.dataset_loader as dl
from esn.reservoir import activation, initialization
from esn.esn import DeepESN
from esn.reservoir.initialization import CompositeInitializer
from esn.reservoir.util import NRMSELoss

norm = True
temp = dl.loader_explicit('datasets/temperature_day.csv', test_size=600)
nrmse = NRMSELoss()

if norm:
    X, X_test, y, y_test, centr, spread = dl.norm_loader__(temp)
    y_test = spread * y_test + centr
else:
    X, X_test, y, y_test = temp()

esn = DeepESN(
    hidden_size=300,
    num_layers=2,
    activation=activation.tanh(leaky_rate=0.7),
    initializer=initialization.WeightInitializer(
        weight_hh_init=CompositeInitializer() \
            .spectral_noisy()
            .spectral_normalize() \
            .scale(factor=0.9)
    )

)
esn.fit(X, y)

if norm:
    output = spread * esn(X_test) + centr
else:
    output = esn(X_test)

n = nrmse(output.unsqueeze(-1), y_test).item()
print(n)
last = 100
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()
