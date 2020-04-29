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

esn = DeepESN(1,100,initialization=default_init,num_layers=1,activation=A.tanh,transient=365)
# print(esn(X).size())
esn.fit(X,y)

from matplotlib import pyplot as plt
plt.plot(range(378),esn(X_test).view(-1).detach().numpy(),'r')
plt.plot(range(378),y_test.view(-1).detach().numpy(),'b')
plt.show()
plt.plot(range(20),esn(X_test).view(-1).detach().numpy()[-20:],'r')
plt.plot(range(20),y_test.view(-1).detach().numpy()[-20:],'b')
plt.show()
# print(esn(stock_loader()[1]).size())
# print(esn(stock_loader()[1])[5])
