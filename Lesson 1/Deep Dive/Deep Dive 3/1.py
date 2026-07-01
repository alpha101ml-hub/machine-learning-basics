import numpy as np

# From Simple to advanced
# 1. Dot product (1D vectors)

a = np.array([1,2,3])
b = np.array([4,5,6])
result = np.einsum('i,i->', a, b)  # -> scalar
# Equivalent to np.dot(a,b) or np.sum(a*b)

# Matrix multiplication

A = np.random.rand(3,4)
B = np.random.rand(4,5)
# 'i' (rows of A), 'k' (cols of A / rows of B), 'j' (cols of B)
# 'k' appears in inputs but not output -> summed over

# Trace (sum of diagonal)

M = np.random.rand(5,5)
trace = np.einsum('ii->', M)  # same as np.trace(M)

# Outer product

a = np.array([1,2,3])
b = np.array([4,5])
np.outer(a, b) == np.einsum('i,j->ij', a, b)
# No sum - all indices appear in output -> elementwise product grid