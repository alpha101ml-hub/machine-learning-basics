# Create a 2D matrix X of shape (5,4) with random integers between 0 and 10

import numpy as np

X = np.random.randint(0, 11, size=(5,4))
# np.random.randint(low, high, size): This function generates random integers

print("Matrix X:")
print(X)
print("Shape:", X.shape) # matix of shape (5,4) - (row, column)


# Create a column vector c of shape (5,1) filled with random integers between 0 and 10

import numpy as np
c = np.random.randint(0, 11, size=(5,1))

print("Column vector c:")
print(c)
print("Shape:", c.shape)


# Add c to each column of X using broadcasting - save as Y

import numpy as np

# Assuming X is (5,4) and c is (5,1) from the previous steps
# X = np.random.randint(0, 11, size=(5,4))
# c = np.random.randint0, 11, size(5,1))

# Add c to X using broadcasting
Y = X + c

print("Matirx X (5,4):")
print(X)

print("\nColumn Vector c (5,1:)")
print(c)

print("\nResulting Matrix Y (X + c):")
print(Y)
print("Shape of Y:", Y.shape)

# Compute the dot product of each row of Y with itself (i.e., row-wise squared norm) using one line of vectorized code (no loop)

# Compute the row-wise squared noem of Y using einsum
row_norms = np.einsum('ij, ij->i', Y, Y)
# Or alternatively, using element-wise multiplication and summing across columns:
# row_norms = np.sum(Y**2, axis=1)

print("Row-wise squared norms:", row_norms) # Y0*Y0 + Y1*Y1 + Y2*Y2 + Y3*Y3
print("Shape:", row_norms.shape)

# Compute the same result using einsum.

import numpy as np

# Compute row-wise squared norms using einsum
row_norms = np.einsum('ij, ij->i', Y, Y)

print("Result using einsum:", row_norms)


# Time both methods for a large matrix of size (1000, 100) - is the difference significant?

import time
import numpy as np

# Create the large matrix
Y = np.random.randint(0, 11, size=(1000, 100)).astype(float)

iterations = 500

# 1. Timing np.sum(y**2, axis=1)
start = time.perf_counter()
for _ in range(iterations):
    res_sum = np.sum(Y**2, axis = 1)
end = time.perf_counter()
time_sum = (end - start) / iterations

# 2. Timing np.einsum('ij, ij->i', Y, Y)
start = time.perf_counter()
for _ in range(iterations):
    res_einsum = np.einsum('ij, ij->i', Y, Y)
end = time.perf_counter()
time_einsum = (end - start) / iterations

print(f"time for np.sum over {iterations:,} runs: {time_sum:.4f} seconds")
print(f"time for np.einsum over {iterations:,} runs: {time_einsum:.4f} seconds")