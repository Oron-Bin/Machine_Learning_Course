
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

##################################################################
### Load and perpare data ###

# training data
with open('auto_kml.pkl', 'rb') as H:
    data = pickle.load(H)

# Input: Features of various cars
# 0. displacement
# 1. Number of cylinders
# 1. horsepower
# 2. weight
# Output: Fuel consumption (km/l)

X = data['features']
Y = data['kml'].reshape(-1,1)
D = np.concatenate((X,Y), axis=1)

scaler = StandardScaler()
scaler.fit(D)
D = scaler.transform(D)

X_train, X_test, y_train, y_test = train_test_split(D[:,:-1], D[:,-1], test_size=0.15, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

##################################################################
### Build model ###

model = nn.Sequential(
    nn.Linear(X_train.shape[1], 3),
    nn.Tanh(),
    nn.Linear(3, 3),
    nn.Tanh(),
    nn.Linear(3, 1),
    nn.Tanh())

print(model)

##################################################################
### Preparare for Training ###

loss_fn = nn.MSELoss()  # Mean Square Error

# The optimizer is an algorithm used to adjust the model weights progressively 
# to produce a better output. 
# There are many types of optimizers to choose.
# Adam optimizer is a popular version of gradient descent that can automatically 
# tune itself and gives good results in a wide range of problems.
# model.parameters() are the weights of the defined model that Adam will optimize.
optimizer = optim.Adam(model.parameters(), lr=0.001) # <- Include learning rate (lr)

epochs = 300
batch_size = 100

##################################################################
### Train ###


L, L_test = [], [] # Record losses
for epoch in range(epochs):

    for i in range(0, len(X_train), batch_size):
        Xbatch = X_train[i:i+batch_size,:]
        y_pred = model(Xbatch) # Feed-forward
        ybatch = y_train[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch) # Evaluate loss
        optimizer.zero_grad() # Zero the gradients before running the backward pass. This is because by default, gradients are accumulated in buffers ( i.e, not overwritten)
        loss.backward() # Compute gradient of the loss with respect to all the learnable parameters of the model
        optimizer.step() # Update weights

    L_test.append(loss_fn(model(X_test), y_test).detach().numpy())
    L.append(loss.detach().numpy())
    print(f'{epoch} - Finished epoch: {epoch}, latest loss: {loss}, test loss: {L_test[-1]}')

# plt.figure(figsize=(8,4))
# plt.plot(np.arange(0, epochs), L, label='Train loss')
# plt.plot(np.arange(0, epochs), L_test, label='Test loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.xlim([0,epochs])
# plt.legend()
# plt.show()
#
# ##################################################################
# ### Make prediction ###
#
# # For one sample
print()
x_query = X_test[5,:]
prediction = model(x_query)

d = np.concatenate((x_query.detach().numpy(), y_test[5].detach().numpy()), axis=0)
d = scaler.inverse_transform(d)
print('For query sample: ', d[:4], ' and label: ', d[-1])

d = np.concatenate((x_query.detach().numpy(), prediction.detach().numpy()), axis=0)
d = scaler.inverse_transform(d)
print('Prediction is: ', d[-1])

# For a batch
y_pred = model(X_test[100:200,:])
print(f"Predictions of batch are: {y_pred}")

