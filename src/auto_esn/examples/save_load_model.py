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

# now choose activation and configure it
activation = self_normalizing_default(spectral_radius=100)

# initialize the esn
esn_original = GroupedDeepESN(
    groups=2,  # choose number of groups
    num_layers=(3, 3),  # choose number of layers for each group
    hidden_size=80,  # choose hidden size for all reservoirs
    activation=activation  # assign activation
)

# fit
esn_original.fit(X, y)

# save model
with open('esn_model.pkl', 'wb') as fn:
    torch.save(esn_original, fn)

# now load it
with open('esn_model.pkl', 'rb') as fn:
    esn_loaded = torch.load(fn)

# predict with original model
output_original = esn_original(X_test)

# predict on loaded model
output_from_loaded_model = esn_loaded(X_test)

# Make sure the output is the same:
print(f"Output's are the same: {torch.equal(output_from_loaded_model, output_original)}")

# evaluate
loaded_err = nrmse(output_from_loaded_model, y_test).item()
original_err = nrmse(output_original, y_test).item()
print(f"Error for loaded model: {loaded_err}")
print(f"Error for original model: {original_err}")

# plot
last = 200
plt.plot(range(last), output_from_loaded_model.view(-1).detach().numpy()[-last:], 'r', label='loaded')
plt.plot(range(last), output_original.view(-1).detach().numpy()[-last:], 'g', label='original')
plt.plot(range(last), y_test.view(-1).detach().numpy()[-last:], 'b', label='truth')
plt.legend()
plt.show()
