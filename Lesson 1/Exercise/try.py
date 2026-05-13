import numpy as np

A = np.random.rand(3,4)
B = np.random.rand(4,5)
C = np.einsum('ik,kj->ij', A, B)   # same as A @ B

print(A)
print(B)
print(C)