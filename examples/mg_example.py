

import esn.activation as A
import utils.dataset_loader as dl
from esn.esn import DeepESNCell, ESNBase, SVDReadout
from esn.initialization import WeightInitializer, default_hidden
from esn.util import NRMSELoss
from matplotlib import pyplot as plt

norm =True
size =200
layers =2
bias = False
leaky = 0.8
act_radius = 500.
activation = A.self_normalizing_default(leaky_rate=leaky, spectral_radius=act_radius)
include_input=True
mg17clean = dl.loader_explicit('datasets/mg.csv',2_000,6_000)
transient=30
nrmse = NRMSELoss()


if norm:
    X, X_test, y, y_test, centr, spread = dl.norm_loader__(mg17clean)
    y_test = spread * y_test + centr
else:
    X, X_test, y, y_test = mg17clean()
initializer = WeightInitializer(weight_hh_init = default_hidden(spectral_radius=1.0))
_esn = DeepESNCell(1, size, initializer= initializer,num_layers=layers, bias=bias,
                            activation=activation, include_input=include_input)
esn = ESNBase(reservoir=_esn,
              readout=SVDReadout(total_hidden_size=sum([l.hidden_size for l in _esn.layers]),output_dim=1),transient=transient)
esn.fit(X, y)

if norm:
    # output = spread * output + centr
    output = spread * esn(X_test) + centr
else:
    output = esn(X_test)

n = nrmse(output.unsqueeze(-1), y_test).item()
print(n)
last =50
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()