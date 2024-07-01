import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import pickle

# Load oceanic dataset
# Data is normalized
data = pickle.load(file=open('dataset/dataset.pkl', "rb"))
X = data['data'] # [depth, water temperature]
Y = data['target'] # [Salinity of water]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def Loss(y, y_pred): 
    M = len(y)
    S = 0
    for i in range(M):
        S += (y[i] - y_pred[i])**2
    return (1/M) * S

#derivative of loss w.r.t weight
def dLoss_dW(x, y, y_pred):
    M = len(y)
    S = 0
    for i in range(M):
        S += -x[i] * (y[i] - y_pred[i])
    return (2/M) * S

# code for "wx+b"
def predict(W, X):
    Y_p = []
    for x in X:
        Y_p.append(W.dot(x))
    return np.array(Y_p)

# Get random batch for Stochastic GD
def get_batch(X, y, batch_size = 500):
    ix = np.random.choice(X.shape[0], batch_size)
    return X[ix, :], y[ix]

Wu = np.random.randn(X.shape[1]) # Initial weigth vector
learning_rate = 0.01
epochs = 1000

SGD = False # Use Stochastic GD?

L_train = []
L_test = []
for i in range(epochs):
     # Get data
    if not SGD:
        X_batch, y_batch = X_train, y_train
    else:
        X_batch, y_batch = get_batch(X_train, y_train)
    
    Y_p = predict(Wu, X_batch)
    Wu = Wu - learning_rate * dLoss_dW(X_batch, y_batch ,Y_p)  # Update weights

    L_train.append(Loss(y_batch, Y_p))
    L_test.append(Loss(y_test, predict(Wu, X_test)))

print('Weights: ', Wu)
print('Train loss: ', L_train[-1])
print('Test loss: ', L_test[-1])

###### Plot learning curve ######

plt.figure()
plt.plot(L_train, '-k', label = 'Train loss')
plt.plot(L_test, '-r', label = 'Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

##### Plot fitted plane ######

res = 20
x = X_test[:, 0]
y = X_test[:, 1]
z = y_test

x_s = np.linspace(min(x), max(x), res)
y_s = np.linspace(min(y), max(y), res)
x_s, y_s = np.meshgrid(x_s, y_s)

z_s = np.zeros((len(x_s), len(y_s)))
for i in range(len(x_s)):
    for j in range(len(y_s)):
        z_s[i, j] = predict(Wu, np.array([x_s[i,j], y_s[i,j]]).reshape(1,-1))

fig = plt.figure(2, figsize=(10,7))
ax = plt.axes(projection='3d')

ax.plot_surface(x_s, y_s, z_s, cmap=cm.hot, alpha=0.3, linewidth=0.2, edgecolors='black')
ax.scatter(x, y, z, label='Test data')
ax.scatter(X_train[:, 0], X_train[:, 1], y_train, label='Train data')
ax.legend()

plt.show()