import utils.dataset_loader as dl
import numpy as np
from esn.esn import DeepESN
from esn import activation as A
stock_loader = dl.loader('../datasets/stock_price_day.csv', 0.7, 10_000)
from esn.initialization import *


from sklearn.linear_model import LinearRegression, Ridge

#sequence length, batch size, features
#torch.Size([880, 1, 1])
print(stock_loader()[0].size())
print(stock_loader()[1].size())
X,X_test,y,y_test = stock_loader()
X,X_test,y,y_test = X - torch.mean(X),X_test-torch.mean(X_test),y-torch.mean(y),y_test-torch.mean(y_test)
esn = DeepESN(1,365,initialization=WeightInitializer(),num_layers=3,bias=False,activation=A.tanh(leaky_rate=0.9),transient=365)
# print(esn(X).size())
esn.fit(X,y)

from matplotlib import pyplot as plt
plt.plot(range(378),esn(X_test).view(-1).detach().numpy(),'r')
plt.plot(range(378),y_test.view(-1).detach().numpy(),'b')
plt.show()
plt.plot(range(50),esn(X_test).view(-1).detach().numpy()[-50:],'r')
plt.plot(range(50),y_test.view(-1).detach().numpy()[-50:],'b')
plt.show()
# print(esn(stock_loader()[1]).size())
# print(esn(stock_loader()[1])[5])
