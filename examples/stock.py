import utils.dataset_loader as dl
import numpy as np
from esn.esn import DeepESN, DeepSubreservoirESN
from esn import activation as A

stock_loader = dl.loader('../datasets/stock_price_day.csv', 0.7, 10_000)
from esn.initialization import *

from sklearn.linear_model import LinearRegression, Ridge

# sequence length, batch size, features
# torch.Size([880, 1, 1])
print(stock_loader()[0].size())
print(stock_loader()[1].size())
X, X_test, y, y_test = stock_loader()
# X, X_test, y, y_test = X - torch.mean(X), X_test - torch.mean(X_test), y - torch.mean(y), y_test - torch.mean(y_test)
# X, X_test, y, y_test = X, X_test, y, y_test
# esn = DeepESN(1, 365, initializer=WeightInitializer(), num_layers=3, bias=False, activation=A.tanh(leaky_rate=0.6),
#               transient=365)
# #todo spr self normalizing
# # print(esn(X).size())
# # esn.fit(X, y)
#
#
from matplotlib import pyplot as plt
#
# pred = esn(X_test)
# plt.plot(range(378), pred.view(-1).detach().numpy(), 'r')
# plt.plot(range(378), y_test.view(-1).detach().numpy(), 'b')
# plt.show()
# plt.plot(range(50), pred.view(-1).detach().numpy()[-50:], 'r')
# plt.plot(range(50), y_test.view(-1).detach().numpy()[-50:], 'b')
# plt.show()
# pred = esn(X_test)
# plt.plot(range(378), pred.view(-1).detach().numpy(), 'r')
# plt.plot(range(378), y_test.view(-1).detach().numpy(), 'b')
# plt.show()


esn = DeepSubreservoirESN(1,1, initializer=SubreservoirWeightInitializer(subreservoir_size=30), num_layers=3, bias=False, activation=A.tanh(leaky_rate=0.6),
              transient=365)
#todo spr self normalizing
# print(esn(X).size())
esn.fit(X, y)
pred = esn(X_test)
plt.plot(range(378), pred.view(-1).detach().numpy(), 'r')
plt.plot(range(378), y_test.view(-1).detach().numpy(), 'b')
# plt.show()

esn.reset_hidden()
esn.grow()
esn.fit(X, y)
pred = esn(X_test)
plt.plot(range(378), pred.view(-1).detach().numpy(), 'y')
# plt.plot(range(378), y_test.view(-1).detach().numpy(), 'b')
# plt.show()
esn.reset_hidden()
esn.grow()

# plt.plot(range(50), pred.view(-1).detach().numpy()[-50:], 'r')
# plt.plot(range(50), y_test.view(-1).detach().numpy()[-50:], 'b')
# plt.show()
esn.fit(X, y)
pred = esn(X_test)
plt.plot(range(378), pred.view(-1).detach().numpy(), 'g')
# plt.plot(range(378), y_test.view(-1).detach().numpy(), 'b')
plt.show()
