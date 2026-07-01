# The row sums of a 2D matrix (hint: 'ij->i')

import numpy as np

a = np.array([[1,2,3],
              [4,5,6]])

b = np.einsum('ij->i', a)  # -> shape (2,)
print(b)

# The column sums (hint: 'ij->j')

c = np.einsum('ij->j', a)
print(c)  # -> shape (3,)

# Matrix transpose ('ij->ji')

d = np.einsum('ij->ji', a)
print(d)  # -> shape (3,2)

# Batch matrix multiplication – two 3D arrays of shape (batch, m, n) and (batch, n, p)

e = np.random.rand(10, 3, 4)  # batch of 10 matrices of shape (3,4)
f = np.random.rand(10, 4, 5)  # batch of 10 matrices of shape (4,5)
g = np.einsum('bij,bjk->bik', e, f) 
print(g.shape)  # -> shape (10, 3, 5)