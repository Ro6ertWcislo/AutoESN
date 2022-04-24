import random

import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, Tensor
from sklearn.linear_model import Ridge

auto_esn_path = r"D:\\program_files_d\\python\\AutoESN\\src"
sys.path.append(auto_esn_path)

from auto_esn.esn.esn import DeepESN, FlexDeepESN
from auto_esn.esn.reservoir import activation, initialization
from auto_esn.esn.readout.nn_readout import AutoNNReadout, ReadoutMode
from collections import Counter
np.random.seed(42)

def create_dataloader(X, y, batch_size):
    data = []
    for i in range(len(X)):
        data.append([X[i], y[i]])
    loader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)
    return loader

# LOADING DATA
with open(r"D:\program_files_d\python\master\notebooks\dataset\alco_dataset/balanced_dataset.p", 'rb') as file:
    data = pickle.load(file)

# LOADING DATA...
X = data['X']
y = data['y']


count_val = [i[0] for i in y]
print("ALL DATA: ", Counter(count_val))

# SPLITTING DATA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)


# CONVERTING DATA TO TORCH TENSORS
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)
y_train = torch.Tensor(y_train)
y_test = torch.Tensor(y_test)

# PREPARE HYPER-PARAMETERS
leaky_rate = 0.8
activation = activation.tanh(leaky_rate)
hidden_size = 1000
num_layers = 3
regularization = 1
radius = 0.9
input_scaling = 0.9
#initializer = initialization.WeightInitializer(radius=radius, input_scaling=input_scaling)
initializer = initialization.CustomWeightInitializer(radius=radius, input_scaling=input_scaling)
batch_size = 32
learning_rate = 1e-5

# # PREPARE HYPER-PARAMETERS
# leaky_rate = float(sys.argv[1])
# activation = activation.tanh(leaky_rate)
# hidden_size = int(sys.argv[2])
# num_layers = int(sys.argv[3])
# regularization = 1
# radius = float(sys.argv[4])
# input_scaling = float(sys.argv[5])
# initializer = initialization.WeightInitializer(radius=radius, input_scaling=input_scaling)
# batch_size = int(sys.argv[6])
# learning_rate = 1e-5

print(f"{leaky_rate}, {hidden_size}, {num_layers}, {radius}, {input_scaling}, {batch_size}")

# PREPARE ESN MODEL
esn = FlexDeepESN(
        input_size=64,
        num_layers=num_layers,
        hidden_size=hidden_size,
        activation=activation,
        initializer=initializer,
        readout=AutoNNReadout(input_dim=num_layers * hidden_size * 256,
                              lr=learning_rate,
                              epochs=30,
                              batch_size=batch_size,
                              mode=ReadoutMode.BinaryClassification,
                              params = {'hidden_size': hidden_size, "layers": num_layers}))

# IF CUDA IS AVAILABLE THEN CONVERT TENSORS AND MODEL TO CUDA
CUDA = True
if CUDA:
    CUDA = torch.cuda.is_available()

if CUDA:
    X_train = X_train.cuda()
    X_test = X_test.cuda()
    y_train = y_train.cuda()
    y_test = y_test.cuda()
    esn.to_cuda()

# FITTING MODEL
esn.fit(X_train, y_train)

accs = []
test_loader = create_dataloader(X_test, y_test, 8)
predictions = []
trues = []

for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    output = esn(data)
    #print("TESTING")
    preds = output.reshape(-1).detach().cpu().numpy().round()
    trues.extend(torch.squeeze(target).detach().cpu())
    predictions.extend(preds)

print("RAPORT: ", classification_report(trues, predictions))

