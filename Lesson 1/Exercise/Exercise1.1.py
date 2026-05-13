'''
Exercise 1.1 (Novice): Create two large random arrays of size 1,000,000. Time the loop vs np.dot using time.perf_counter(). Observe the difference.
'''

import numpy as np
import time

s = 1000000
a = np.random.rand(s)
b = np.random.rand(s)


# Manual loop(slow)
def dot_loop(x,y):
    result = 0.0
    for i in range(len(x)):
        result += x[i] * y[i]
    return result


# Vectorised using np.dot (fast)
def dot_vec(x,y):
    return np.dot(x,y) # or x @ y

# time the loop
start = time.perf_counter()
res_loop = dot_loop(a,b)
end = time.perf_counter()
end = time.perf_counter()
print(f"Loop time: {end - start:.6f} seconds")
print(f"Result (loop): {res_loop:.6f}")

# Time the vectorised version
start = time.perf_counter()
res_vec = dot_vec(a,b)
end = time.perf_counter()
print(f"Vectorised time: {end - start:.6f} seconds")
print(f"Result (vectorised): {res_vec:.6f}")
print(f"Result (np.dot): {res_vec:.6f}")

# Verify they give the same answer (within floating point error)
print(f"Diiferenc: {abs(res_loop - res_vec):.6f}")
 