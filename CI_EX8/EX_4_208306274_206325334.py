import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

with open('train_data.pkl', 'rb') as H:
    data_train = pickle.load(H)
X_train = data_train['features']
Y_train = data_train['labels'].reshape(-1,1)


with open('test_data.pkl', 'rb') as H:
    data_test = pickle.load(H)
X_test = data_test['features']
Y_test = data_test['labels'].reshape(-1,1)

# print(len(X_test))

scalerX = StandardScaler()
scalerX.fit(X_train)
X_train_normalized = scalerX.transform(X_train)
X_test_normalized = scalerX.transform(X_test)

scalarY = StandardScaler()
scalarY.fit(Y_train)
Y_train_normalized = scalarY.transform(Y_train)
Y_test_normalized = scalarY.transform(Y_test)

X_train = torch.tensor(X_train_normalized, dtype=torch.float32)
Y_train = torch.tensor(Y_train_normalized, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test_normalized, dtype=torch.float32)
Y_test = torch.tensor(Y_test_normalized, dtype=torch.float32).reshape(-1, 1)

### Build model ###
model =nn.Sequential(
            nn.Linear(26,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.Linear(32,12),
            nn.Linear(12, 1))

# print(model)

### Prepare for Training ###
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

epochs = 300
batch_size = 100

### Train ###
L, L_test = [], []  # Record losses
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i + batch_size, :]
        ybatch = Y_train[i:i + batch_size]

        y_pred = model(Xbatch)  # Feed-forward

        loss = loss_fn(y_pred, ybatch)  # Evaluate loss
        optimizer.zero_grad()  # Zero the gradients before running the backward pass. This is because by default, gradients are accumulated in buffers (i.e., not overwritten)
        loss.backward()  # Compute gradient of the loss with respect to all the learnable parameters of the model
        optimizer.step()  # Update weights

    # Record losses for plotting
    model.eval()
    L_test.append(loss_fn(model(X_test), Y_test).item())
    L.append(loss.item())
    print(f'{epoch} - Finished epoch: {epoch}, train loss: {loss}, val loss: {L_test[-1]}')

# Calculate the final training and test losses using the not-normalized data
with torch.no_grad():
    model.eval()
    k_best = SelectKBest(score_func=f_regression, k=26)
    k_best.fit(X_train,Y_train.ravel())
    feature_scores=k_best.scores_
    feature_scores_rounded=[round(num,2) for num in feature_scores]
    # Revert the normalized data back to its original scale
    # X_train_denormalized = scalerX.inverse_transform(X_train_normalized)
    # X_test_denormalized = scalerX.inverse_transform(X_test_normalized)

    Y_train_pred_denormalized = model(torch.tensor(X_train_normalized, dtype=torch.float32))
    Y_test_pred_denormalized = model(torch.tensor(X_test_normalized, dtype=torch.float32))

    Y_train_pred = scalarY.inverse_transform(Y_train_pred_denormalized)  # Convert tensor to a NumPy array
    Y_test_pred = scalarY.inverse_transform(Y_test_pred_denormalized)  # Convert tensor to a NumPy array

    # print(type(Y_train_pred))
    # print(type(Y_train.numpy()))
    # Calculate the RMSE using the not-normalized data
    rmse_train = np.sqrt(((Y_train_pred - Y_train.numpy()) ** 2).mean())
    rmse_test = np.sqrt(((Y_test_pred - Y_test.numpy()) ** 2).mean())

# print("Final Training RMSE (not-normalized):", rmse_train)
print("Final Test RMSE (not-normalized):", rmse_test)

print("Feature Scores:", feature_scores_rounded)