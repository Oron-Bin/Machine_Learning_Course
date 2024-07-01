
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------
# plotting - 2D

x = np.linspace(0, 100, 1000)
y1 = x**2 / 100
y2 = -x + 3

f = plt.figure(figsize=(10,5))
plt.plot(x, y1, '-k', label = 'x**2 / 100', linewidth = 2)
plt.plot(x, y2, ':b', label = '-x + 3', linewidth = 1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Functions')
plt.legend()

f, ax = plt.subplots(1, 2, figsize=(6,5))#, sharey=True)
ax[0].plot(x, y1, '-k', linewidth = 2)
ax[0].set_title('Function 1')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].plot(x, y2, '-k', linewidth = 2)
ax[1].set_title('Function 2')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

plt.show()


# ----------------------------------------------
# plotting - 2D

xs = np.array([1,1])
xg = np.array([8,10])
S = np.linspace(0, 1, 20)   # line = xs + s * (xg-xs)
X = np.array([xs + s * (xg-xs) for s in S])

circ = [8,4]
ro = 2
pts = np.array([[0,2], [4,5], [1,4.5]])

fig, ax = plt.subplots()
ax.plot(X[:,0], X[:,1], '-s')
circle = plt.Circle((circ[0], circ[1]), ro, color='g')
rec = plt.Rectangle((2,7), 3, 2, fc='gray',ec="red")
p = plt.Polygon(pts, fc = 'magenta')

ax.add_artist(circle)
ax.add_artist(rec)
ax.add_artist(p)

plt.axis('equal')

plt.show()

# ----------------------------------------------
# plotting - 3D
from mpl_toolkits import mplot3d

r = 2.0
P = []
for _ in range(100000):
    
    # a -> b
    # np.random.random() * (a-b) + b
    theta = np.random.random() * (2 * np.pi) - np.pi
    phi = np.random.random() * 2 * np.pi + 0

    p = np.array([r * np.sin(theta) * np.cos(phi), 
                  r * np.sin(theta) * np.sin(phi), 
                  r * np.cos(theta)])
    P.append(p)
P = np.array(P)
print(P.shape)

Z = np.linspace(-2, 2, 1000)
X = np.sin(14*Z)
Y = np.cos(14*Z)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(P[:,0], P[:,1], P[:,2], '.k', markersize = 0.5)
ax.plot3D(X, Y, Z, 'red')

plt.show()


