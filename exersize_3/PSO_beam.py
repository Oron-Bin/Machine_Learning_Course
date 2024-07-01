import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# https://apmonitor.com/me575/uploads/Main/optimization_book.pdf

F = 4000 # N
# L = 1.8 # m
rho = 8000 # Kg/m^3
max_stress = 2100e6# 2617e6 # N/m^2 = Pa
K = 3.0
stop_iter = 50

########### Objective function ###########

def Weight(x):
    L, R, t = x[0], x[1], x[2]
    return np.pi*rho*(R**2 - (R-t)**2)*L + K/L

def Stress(x):
    L, R, t= x[0], x[1], x[2]
    I = (R**4 - (R-t)**4)*np.pi/4
    return F*L*R/I

def generate_valid_particles(n_particles, x_min, x_max):
    X = []
    count = 0
    while count < n_particles:
        x = np.random.rand(ndim, 1) * (x_max - x_min) + x_min
        if Stress(x) < max_stress:
            X.append(x)
            count += 1

    X = np.array(X).reshape(n_particles,-1).T
    return X


def f(X):
    O = []
    for x in X.T:
        O.append(Weight(x))

    return np.array(O)

########### Define parameters ###########
# x = [L, R, t]
x_min = np.array([0.5, 12e-3, 1e-3]).reshape(-1,1) # m, m, m
x_max = np.array([2.3, 100e-3, 10e-3]).reshape(-1,1) # m, m, m, m, m
ndim = len(x_min)

#### Hyper-parameter of the algorithm ###
c1 = 0.1
c2 = 0.1
w = 0.8

# Create particles
n_particles = 100
np.random.seed(100)
# X = np.random.rand(ndim, n_particles) * (x_max - x_min) + x_min # Random particles within the bounds
X = generate_valid_particles(n_particles, x_min, x_max)
V = np.random.randn(ndim, n_particles) * 0.1 # Random initial velocity

# Initialize data
pbest = X # Initialize personal best as first generation
pbest_obj = f(X)
gbest = pbest[:, pbest_obj.argmin()] # Global best particles
gbest_obj = pbest_obj.min()

plt.figure()
G = []

for j in range(100):
    plt.clf()
    # Update params
    r1, r2 = np.random.rand(2)
    V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
    X_temp = X + V
    for i in range(n_particles): # Check for constraints
        if np.all(X_temp[:,i] > x_min.reshape(-1,)) and np.all(X_temp[:,i] < x_max.reshape(-1,)) and Stress(X_temp[:,i]) < max_stress:
            X[:,i] = X_temp[:,i].copy()
    obj = f(X)
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj) ] # Update for each particle personal best
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
    gbest = pbest[:, pbest_obj.argmin()] # Update global minimum
    gbest_obj = pbest_obj.min()
    print('Iteration', j, ', Best weight so far: ', gbest_obj, '; L=' + str(gbest[0]), 'R=' + str(gbest[1]), 't=' + str(gbest[2]), 'S=' + str(Stress(gbest)*1e-6) + 'MPa')
    G.append(gbest_obj)

    plt.plot(G)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.pause(0.0001)

    if j > stop_iter and np.all(np.array(G[-stop_iter:])==G[-1]):
        break

print()
print('PSO found a solution after %d iterations:'%(j-stop_iter))
print('W = ' + str(round(gbest_obj,3)) + ' Kg')
print('L = ' + str(round(gbest[0],3)) + ' m')
print('R = ' + str(round(gbest[1]*1e3,3)) + ' mm')
print('t = ' + str(round(gbest[2]*1e3,3)) + ' mm')
print('S = ' + str(round(Stress(gbest)*1e-6,3)) + ' MPa')
plt.show()