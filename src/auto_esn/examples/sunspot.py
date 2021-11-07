import time

import auto_esn.utils.dataset_loader as dl
from auto_esn.datasets.df import Sunspot
from auto_esn.esn.esn import DeepESN
from auto_esn.esn.reservoir.initialization import CompositeInitializer, WeightInitializer
from auto_esn.esn.reservoir.util import NRMSELoss
from matplotlib import pyplot as plt
norm = True
sunspot = dl.loader_explicit(Sunspot, test_size=600)
nrmse = NRMSELoss()
if norm:
    X, X_test, y, y_test, centr, spread = dl.norm_loader__(sunspot)
    y_test = spread * y_test + centr
else:
    X, X_test, y, y_test = sunspot()

i = CompositeInitializer()\
    .with_seed(12)\
    .uniform()\
    .regular_graph(4)\
    .spectral_normalize()\
    .scale(0.9)

w = WeightInitializer()
w.weight_hh_init = i

esn = DeepESN(initializer= w, hidden_size=500, num_layers=2)
start = time.time()
# esn.to_cuda()
esn.fit(X, y)

if norm:
    output = spread * esn(X_test) + centr
else:
    output = esn(X_test)
print(time.time()-start)
n = nrmse(output, y_test).item()
print(n)
last = 50
plt.plot(range(last), output.view(-1).detach().numpy()[-last:], 'r')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b')
plt.show()
