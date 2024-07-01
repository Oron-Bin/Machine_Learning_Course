import numpy as np 

# -----------------------------------------------
# Numpy arrays

print("1------------------------------")
a = np.array([1,2,3,4,5])
b = np.array([1,2,3])

print(a, b)
print(a.shape, b.shape)

print("2------------------------------")

a = a.reshape(1,-1)
b = b.reshape(1,-1)
print(a, b)
print(a.shape, b.shape)
print("3------------------------------")

a = a.reshape(-1,)
b = b.reshape(-1,)
print(a, b)
print(a.shape, b.shape)

print("4------------------------------")

c = np.array([[1,2,3],[4,5,6],[7,8,9],[-1,-8,9]])
print(c.shape)
print(c)
print(c.T)

print("5------------------------------")

print(c.dot(c.T)) # c * C.T
print(c.T.dot(c)) # = np.matmul(c.T, c)
print(np.multiply(c, c)) # element wise multiplication

print("6------------------------------")

a = np.array([1,2,3,4,5])
b = np.array([6,7,8,9,10])

a = a.reshape(-1,1)
b = b.reshape(-1,1)
print(a.shape, b.shape)
c = np.concatenate((a,b), axis=0)
print(c)
print(c.shape)

print("7------------------------------")

b = np.copy(a)
# b = a
print(a, b)
b[2] = -1000
print(a)
print(b)

print("8------------------------------")


a1 = np.zeros((10,4))
a2 = np.ones((10,4))
print(a1)
print(a2)

print("9------------------------------")


# -----------------------------------------------
# Numpy - slicing
a = np.array([1,2,3,4,5])
v = a[:2]
v = a[1:3]
v = a[:-3]


c = np.array([[1,2,3],[4,5,6],[7,8,9],[-1,-8,9]])
v = c[:2,:-1]
print(v)

v = c[:,1:3]
print(v)

# -----------------------------------------------
# Numpy - linear algebra
# np.linalg

a = np.array([1,3,4])
b = np.array([9,3,4])
a = np.linalg.norm(a - b) # ||a-b||


print("------------------------------")
# inv_c1 = np.linalg.inv(c[1:,:])
inv_c = np.linalg.pinv(c)
print(inv_c)#, inv_c)

print(np.linalg.det(c[1:,:]))


# -----------------------------------------------
# Numpy - other
print("------------------------------")
a = np.array([1,2,3,4,5])

print(np.sum(a), a.sum())
print(np.mean(a), a.mean())

print(np.std(c, axis=1))


a = np.insert(a, 3, -1)
print(a)
a = np.delete(a, 3)
print(a)

c = np.array([[1,2,3],[4,5,6],[7,8,9],[-1,-8,9]])
b = np.array([9,3,4]).reshape(1,-1)
c = np.insert(c, 2, b, axis=0)
print(c)


th = np.deg2rad(45)
R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0,0,1]])
print(R)

# -----------------------------------------------
# Functions

# def test(a, b, c = 1):
#     d = (a+b)/c
#     if d > 1:
#         return 1
#     else:
#         return 0
#
# k = test(1, 1.2, 3)
# k = test(1, 1.2)

# ===============================================


