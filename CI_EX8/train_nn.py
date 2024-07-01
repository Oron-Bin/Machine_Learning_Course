import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

   
class FCLayer():
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(output_size, input_size) - 0.5
        self.bias = np.random.rand(1, output_size,) - 0.5

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output.reshape(-1,)

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(self.weights.T, output_error) # dL/dx
        weights_error = np.dot(output_error.reshape(-1,1), self.input.reshape(1,-1))

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
    
# -----------------------------------------------------

# Implement tanh activation function
class ActivationLayer():
    def __init__(self):
        self.input = None
        self.output = None
        self.activation = lambda x: np.tanh(x)
        self.activation_prime = lambda x: 1-np.tanh(x)**2

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # No "learnable" parameters ->  no learning rate
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error

# -----------------------------------------------------

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # Loss function 
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2)) # np.power is faster
    
    # Loss function and its derivative
    def mse_prime(self, y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        E = []
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.mse(y_train[j], output)

                # backward propagation
                error_grad = self.mse_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error_grad = layer.backward_propagation(error_grad, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i+1, epochs, err))
            E.append(err)

            plt.figure(0)
            plt.cla()
            plt.plot(range(len(E)), E)
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.pause(0.000001)


if __name__ == "__main__":
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

    # Create network
    net = Network()
    net.add(FCLayer(X_train.shape[1], 3))
    net.add(ActivationLayer())
    net.add(FCLayer(3, 3))
    net.add(ActivationLayer())
    net.add(FCLayer(3, 3))
    net.add(ActivationLayer())
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer())
    
    # Train
    net.fit(X_train, y_train, epochs=100, learning_rate=0.2)
    print()
    predictions = net.predict(X_train)
    err = 0
    for y_p, y in zip(predictions, y_train):
        err += net.mse(y, y_p)
    print('Train loss: ', err/len(y_train))

    # Test
    predictions = net.predict(X_test)
    err = 0
    for y_p, y in zip(predictions, y_test):
        err += net.mse(y, y_p)
    print('Test loss: ', err/len(y_test))
