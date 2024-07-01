import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

##################################################################
### Load and perpare data ###
data = load_digits()

# Show example of 10 images from the dataset
plt.figure(figsize=(10,5))
plt.gray()
I = data.images[0]
for i in range(1,9):
    I = np.concatenate((I, data.images[i]), axis = 1)
plt.imshow(I)
plt.show()

# Flatten the 8x8 image to a vector of length 64
X, y = [], np.array(data.target)
for x in data.images:
    x = x.reshape((-1,))
    X.append(x)
X = np.array(X)

scaler = StandardScaler().fit(X)
X = scaler.transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train).to(torch.int64)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test).to(torch.int64)

##################################################################
### Build model ###

# The nn.Module class is a more flexible way to define a model. It allows you to define a model with any arbitrary structure, including multiple branches and loops.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 32)  
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x): # We don't really need so 3 layers, but for the example.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

model = Net()

##################################################################
### Preparare for Training ###

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 130
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

plt.figure(figsize=(8,4))
plt.plot(np.arange(0, epochs), L, label='Train loss')
plt.plot(np.arange(0, epochs), L_test, '--', label='Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xlim([0,epochs])
plt.legend()
plt.show()

##################################################################
### Evaluate the model ###

correct = 0
for x, y in zip(X_test, y_test):
    output = model(x.reshape(1,-1))
    _, predicted = output.max(1)
    correct += (predicted == y).sum().item()

# Print the accuracy
accuracy = correct / X_test.shape[0] * 100
print('Classification success rate: {}%'.format(accuracy))
