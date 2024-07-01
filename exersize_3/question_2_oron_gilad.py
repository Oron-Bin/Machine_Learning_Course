
'''
question 2 - PSO part
First, we defines the cost function and the constrains include the boundries of each parameter.
After that,, we generate randomly valid particles that stands in the conditions.
this particles appends to a list that calculate their cost func.
In every generation we calculate the global best and the local best, and then applied the pso algorithm.
Finally we get the best solution.
In the end we plot the best sol over the generation

question 2 - ABS part:


'''
import numpy as np
import matplotlib.pyplot as plt


def cost(x):
    T_S, T_h, R, L = x[0], x[1], x[2] ,x[3]
    cost_func = (0.6224*T_S*R*L) + (1.7787*T_h*(R**2)) + (3.1661*(T_S**2)*L+19.84*(T_S**2)*R)
    return cost_func

def constrains(x):

    g1=-x[0]+0.0193*x[2]
    g2 =-x[1]+0.00954*x[2]
    g3 =-(np.pi)*(x[2])*(x[2])*(x[3]) - ((4/3)*(np.pi)*(x[2])**3)+ 1296000
    g4 =x[3] - 240

    if g1 <=0 and g2 <= 0 and g3 <= 0 and g4 <= 0 :
        return True
    return False

def generate_valid_particles(n_particles, x_min, x_max):
    X = []
    count = 0
    while count < n_particles:
        x = np.random.rand(ndim, 1) * (x_max - x_min) + x_min
        if constrains(x) is True:
            X.append(x)
            count += 1

    X = np.array(X).reshape(n_particles,-1).T
    return X

def f(X):
    O = []
    for x in X.T:
        O.append(cost(x))

    return np.array(O)

########### Define parameters ###########
# x = [T_s, T_h, R,L] [inch]
x_min = np.array([0.0001, 0.0001, 10 ,10]).reshape(-1,1) #inch
x_max = np.array([99, 99, 200, 200]).reshape(-1,1) #inch
ndim = len(x_min)
# print(f"shape is like: \n {x_min}")
stop_iter = 50


#### Hyper-parameter of the algorithm ###
c1 = 0.1
c2 = 0.1
w = 0.75

n_particles = 100
np.random.seed(100)
X = generate_valid_particles(n_particles, x_min, x_max)
V = np.random.randn(ndim, n_particles) * 0.1 # Random initial velocity

# Initialize data
pbest = X # Initialize personal best as first generation
pbest_obj = f(X) #cost

gbest = pbest[:, pbest_obj.argmin()] # Global best particle
gbest_obj = min(pbest_obj)

plt.figure()
G = []

for j in range(100):
    plt.clf()
    # Update params
    r1, r2 = np.random.rand(2)

    # V = inertia + cognitive + social
    V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest.reshape(-1, 1) - X)

    # X_now = X_old + V
    X_temp = X + V

    for i in range(n_particles):  # Check for constraints
        if np.all(X_temp[:, i] > x_min.reshape(-1, )) and np.all(X_temp[:, i] < x_max.reshape(-1, )) and constrains(X_temp[:, i]) is True:
                # X_temp[:, i]) < max_stress:
            X[:, i] = X_temp[:, i].copy()

            # X = X_temp.copy()

    # calculate the weights of the updated particles
    obj = f(X)

    # calculate the pbest and gbest
    pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]  # Update for each particle personal best
    pbest_obj = np.array([pbest_obj, obj]).min(axis=0)

    gbest = pbest[:, pbest_obj.argmin()]  # Update global minimum
    gbest_obj = pbest_obj.min()
    print('Iteration', j, ', Best cost so far: ', gbest_obj, '; T_s=' + str(gbest[0]), 'T_h=' + str(gbest[1]),
          'R=' + str(gbest[2]), 'L=' + str(constrains(gbest) * 1e-6) )
    G.append(gbest_obj)

    plt.plot(G)
    plt.ylabel('Cost')
    plt.xlabel('Iterations')
    plt.pause(0.001)

    if j > stop_iter and np.all(np.array(G[-stop_iter:]) == G[-1]):
        break

# print()
print('PSO found a solution after %d iterations:' % (j))
print('f_cost =' , gbest_obj)
print('T_s =' ,str(gbest[0]), 'inch')
print('T_h =' ,str(gbest[1]) , 'inch')
print('R =' ,str(gbest[2]),'inch')
print('L =' ,str(gbest[3]),'inch')
print('g =',-gbest[0]+0.0193*gbest[2], -gbest[1] + 0.00954 * gbest[2],-(np.pi) * (gbest[2]) * (gbest[2]) * (gbest[3]) - ((4 / 3) * (np.pi) * (gbest[2]) ** 3) + 1296000,gbest[3] - 240 )
plt.show()

