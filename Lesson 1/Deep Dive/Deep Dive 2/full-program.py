import time
import numpy as np
import matplotlib.pyplot as plt

# 1. Pure Python Loop Function
def dot_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

# Define the array the sizes requested
sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

times_loop = []
times_numpt_sum = []
times_numpt_dot = []

for size in sizes:
    print(f"Testing size: {size:,} elements...")
    
    # Pure Python data (lists)
    py_a = list(range(size))
    py_b = list(range(size))
    
    # NumPy data (arrays)
    np_a = np.array(py_a, dtype=float)
    np_b = np.array(py_b, dtype=float)
    
    # --- Method 1: Pure Python Loop ---
    start = time.perf_counter()
    _ = dot_loop(py_a, py_b)
    end = time.perf_counter()
    times_loop.append(end - start)
    
    # --- Method 2: NumpPy Vectorized sum(a * b) ---
    start = time.perf_counter()
    _ = np.sum(np_a * np_b)
    end = time.perf_counter()
    times_numpt_sum.append(end - start)
    
    # --- Method 3: NumpPy Optimized Dot ('np.dot') ---
    start = time.perf_counter()
    _ = np.dot(np_a, np_b)
    end = time.perf_counter()
    times_numpt_dot.append(end - start)
    
# Plotting the results
plt.figure(figsize=(10, 6))

plt.plot(sizes, times_loop,marker='o', label='Pure Python Loop', color='red',linewidth=2)
plt.plot(sizes, times_numpt_sum, label='NumPy sum(a * b)', marker='o', color='orange',linewidth=2)
plt.plot(sizes, times_numpt_dot, label='NumPy dot', marker='o', color='green',linewidth=2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size (log scale)', fontsize=12)
plt.ylabel('Time (seconds, log scale)', fontsize=12)
plt.title('Performance Comparison of Dot Product Methods', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.tight_layout()
plt.show()