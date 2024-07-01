'''oron binyamin 208306274 gilad dolev 206325334
this code was very similar to the code that we saw in the exercise in class.
first of all, i imported the library and loaded the pickle file that brought to us.
after that i defined all the function that related to GD , SGD, Normal equation.
i defined an error as a variable that can be changed by the user, and then trained my data as we did in class.
then i changed the size of the parameters that effect the Loss: the learning rate, the batch size, the numbers of epochs and the size of the error.
my main conclusion is that for this type of problem that the deiiiiiiita is not so big and the problem is linear in the parameters,
the normal equation always achieve the best Loss because its the analytic solution. in this case, its better to use it but in cases where the deiiiiiiita is big and the time of calaulation may be long,
the GD will good enough.
second conclusion is that the best learning rate that ive got is 0.0001 and the optimal batch size is 200.
in learning rate that bigger of 0.0001 i got a raise of the error to inf, i think that this happened because we didnt define a bias to the funtion of the GD that make the error to be lower.
the numbers of epochs that gave us a low time and low Loss is between 1000 to 10000.
In conclusion, the demands of this exersize accomplished. (oron binyamin - the deeeeeita man)
'''


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import pickle
import time

# Load oceanic dataset
# Data is normalized
data = pickle.load(file=open('wine_red_dataset.pkl', "rb"))

X = data['features'] # ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
Y = data['quality'] # [Quality]
K = data['feature_names'] # Strings of feature names

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def Loss(y, y_pred):
    M = len(y)
    S = 0
    for i in range(M):
        S += (y[i] - y_pred[i])**2
    return ((1/M) * S) #normalized loss func

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
def get_batch(X, y, batch_size=200):
    ix = np.random.choice(X.shape[0], batch_size)
    return X[ix, :], y[ix]

def theta_normal_eq(X,y): #this is for normal_equation
    theta_upgrade = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    return theta_upgrade

Wu = np.random.randn(X.shape[1])  # Initial weigth vector
learning_rate = 0.0001
epochs = 1000

theta_normal = theta_normal_eq(X_train,y_train)
normal_predict = predict(theta_normal,X_train)
normal_train_loss = Loss(y_train, normal_predict)
normal_test_loss = Loss(y_test,predict(theta_normal,X_test))

SGD =False  # Use Stochastic GD?

time0 = time.time()
L_train = []
L_test = []

L_epochs = []
min_epoch = None
error = 0.05
for i in range(epochs):
    if not SGD:
        X_batch, y_batch = X_train, y_train
    else:
        X_batch, y_batch = get_batch(X_train, y_train)

    Y_p = predict(Wu, X_batch)
    # update the weights
    Wu = (Wu - learning_rate * dLoss_dW(X_batch, y_batch, Y_p))

    L_train.append(Loss(y_batch, Y_p))

    L_test.append(Loss(y_test, predict(Wu, X_test)))
    time_cur = time.time()

    if abs(L_test[-1]-normal_test_loss) < error:
        L_epochs.append(i)
        min_epoch = min(L_epochs)

print('Normal weights: ',theta_normal.reshape(-1,1))
print('Normal equation train lost',normal_train_loss)
print('Normal equation test lost',normal_test_loss, '\n')


if SGD == False:
    print('GD:')
    print('Weights: ', Wu.reshape((-1,1)))
    print('Train loss: ', L_train[-1])
    print('Test loss: ', L_test[-1])
    print(f'Training time: {time_cur - time0:.2f}[secs]')

if SGD == True:
    print('SGD:')
    print('Weights: ', Wu.reshape((-1, 1)))
    print('Train loss: ', L_train[-1])
    print('Test loss: ', L_test[-1])
    print(f'Training time: {time_cur - time0:.2f}[secs]')

if min_epoch == None:
    print(f"error between GD and normal > {error}")
else:
    print(f"epochs number to GD convergence is: {min_epoch}")
###### Plot learning curve ######

plt.figure()
plt.plot(L_train, '-k', label = 'Train loss')
plt.plot(L_test, '-r', label = 'Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()



