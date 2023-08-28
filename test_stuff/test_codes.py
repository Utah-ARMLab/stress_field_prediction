import torch
import numpy as np
import mcubes
# X, Y, Z = np.mgrid[:30, :30, :30]
# u = (X-15)**2 + (Y-15)**2 + (Z-15)**2 - 8**2
# print(X.shape, Y.shape, Z.shape)
# print(u.shape)
# print(u)

X, Y = np.mgrid[:10, :10]
u = (X)**2 + (Y)**2
# print(X.shape, Y.shape)
# print(u.shape)
print(X)
print(Y)
print(u)