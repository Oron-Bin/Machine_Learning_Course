import difflib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

with open('train_data.pkl', 'rb') as H:
    data = pickle.load(H)
X_train = data['features']
Y_train = data['labels'].reshape(-1,1)

with open('test_data.pkl', 'rb') as H:
    data = pickle.load(H)
X_test = data['features']
Y_test = data['labels'].reshape(-1,1)
rer=Y_test
def rid(train):
    lst=[]

    for i in range(train.shape[1]):
        x = []

        for j in train:

                x.append(j[i])


        lst.append(np.std(x))


    return lst

def update_train(X_train):
    X_Tr=[]

    for i in X_train:

        i=np.delete(i,[1,3,5,7,9,13,15,17,21])
        X_Tr.append(i)

    return np.array(X_Tr)

def update_test(X_test):
    X_tes = []
    for j in X_test:
        j = np.delete(j, [1,3,5,7,9,13,15,17,21])
        X_tes.append(j)
    return np.array(X_tes)

def RMSE(predicted,real):

    return sqrt(mean_squared_error(real,predicted))
print(rid(X_train))
#X_train=update_train(X_train)
#X_test=update_test((X_test))

scalerX = StandardScaler()
scalerX.fit(X_train)
X_train = scalerX.transform(X_train)


scalarY=StandardScaler()
scalarY.fit(Y_train)
Y_train=scalarY.transform(Y_train)

scaler_x = StandardScaler()
scaler_x.fit(X_test)
X_test = scaler_x.transform(X_test)

scalar_y=StandardScaler()
scalar_y.fit(Y_test)
Y_test=scalar_y.transform((Y_test))


X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(X_train.shape[1], 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1))

    def forward(self, x):
        x = self.regressor(x)
        return x
model=Net()

loss_fn = nn.MSELoss()


optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs =10000
batch_size =6000
LT=[]
L, L_test = [], []
loss=None
err_best=np.inf
predict_best=None
real_best=None
plt.figure()
for epoch in range(epochs):
    model.train(True)
    plt.clf()

    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size,:]
        y_pred = model(Xbatch)

        ybatch = Y_train[i:i+batch_size]

        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():

        L.append(loss.detach().numpy())
        output=(model(X_test)).detach().numpy()

        pred = scalar_y.inverse_transform(output)
        real = scalar_y.inverse_transform(Y_test.detach().numpy())

        predT= scalarY.inverse_transform(model(X_train).detach().numpy())
        realT=scalarY.inverse_transform(Y_train)
        errt=RMSE(predT,realT)
        LT.append(errt)

        err=RMSE(pred.flatten(), real.flatten())

        if err< err_best:
            predict_best=pred.flatten()
            real_best=real.flatten()

        L_test.append(err)
        #print(L_test[-1])

        print(min(L_test))
        plt.plot(L_test,label='Test')
        plt.plot(LT, label='Train')
        plt.ylabel('RMSE Lost')
        plt.xlabel('epochs')
        plt.legend()
        plt.pause(0.001)
print('prediction: ',predict_best[:10])
print('real: ',real_best[:10])
print('best lost is:',min(L_test))

plt.show()