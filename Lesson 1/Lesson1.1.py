# vectorized - a method in computing where operations are applied to entire arrays (vectors) of data at once, rather than looping through individual elements one by one

'''Let's begin – Lesson 1.1: NumPy Deep Dive (Start of Novice Path)
🧠 Intuition – Why NumPy?

Imagine you have 1 million numbers and you want to add 5 to each.

    Python list: you write a loop – the CPU does one addition at a time.

    NumPy array: the CPU does the addition on all numbers simultaneously (vectorised) – 100x faster.

    Analogy: A list is like moving boxes one at a time. A NumPy array is like putting all boxes on a conveyor belt that processes them together.
'''

import numpy as np

# Python list
py_list = [1,2,3,4]

# NumPy array
np_array = np.array([1,2,3,4])

# Add 5 to each element - vectorised
result = np_array + 5
print(result) # [6 7 8 9]

'''
🎯 Broadcasting – The magic rule

What is it?
Operations between arrays of different shapes – NumPy automatically stretches (broadcasts) the smaller array to match the larger one.

Rule: Two dimensions are compatible when they are equal, or one of them is 1.

Example: Add a row vector to every row of a matrix
'''

matrix = np.array([[1,2,3],
                   [4,5,6]])
row_vector = np.array([10,20,30])

result = matrix + row_vector
# row_vec broadcast to shape (2,3) by repeating it vertically
print(result)
#[[11 22 33]
# [14 25 36]]

# Pro tip: Broadcasting avoids wasteful memory allocation (no actual repeat happens).

# Vectorization - Ditch the loops
# Bad (slow) way:

def dot_product_loop(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result

'''The Step-by-Step Logic
Let's use a = [1, 2, 3] and b = [4, 5, 6].
result = 0: We start with a clean slate (our accumulator).
for i in range(len(a)): We look at every index (0, 1, 2).
The Calculation (result += a[i] * b[i]):
    Loop 1 (i=0): 1 x 4 = 4. Result is now 4.
    Loop 2 (i=1): 2 x 5 = 10. Result is 4+10 =14.
    Loop 3 (i=2): 3 x 6 = 18. Result is 14+18=32
return result: The final answer is 32.
'''

# Good (fast) way:

def dot_product_vectorized(a,b):
    return np.sum(a * b) # ot a @ b (matrix multiplication)

'''
Step 1: Element-wise Multiplication (a * b)

When you multiply two NumPy arrays, Python doesn't do "Matrix Multiplication" yet; it does Element-wise multiplication. It pairs them up like couples at a dance.

    It takes the first item of a and multiplies it by the first item of b.

    It takes the second item of a and multiplies it by the second item of b.

    This creates a new, temporary array of the results.
'''

'''
Step 2: The Collapse (np.sum)

Then, the np.sum() function acts like a trash compactor. It takes that new temporary array and adds every single number inside it to produce one final scalar (number).

Feature,for-loop (Your first version),
Speed -Slow (Python talks to the CPU for every single number).
Readability- 4-5 lines of code. 
Errors -Easy to mess up the range or the index.

np.sum(a * b) (Vectorized)
Speed - Lightning Fast (NumPy sends the whole ""block"" to the CPU at once)."
Readability - 1 line of code.
Errors - Harder to mess up (it just works).
'''


'''
📐 Einsum – The ultimate weapon (intermediate → pro)

einsum looks scary but it's just a way to say “sum over certain indices”.

Example: Matrix multiplication C = A @ B

    A shape (i, k), B shape (k, j) → C shape (i, j).

    Einsum string: "ik,kj->ij"
'''

A = np.random.rand(3,4)
B = np.random.rand(4,5)
C = np.einsum('ik,kj->ij', A, B)   # same as A @ B


'''
Going pro: You can do batched dot products, trace, outer product with the same notation.

Mastery exercise: Implement np.matmul, np.trace, and np.diag using only einsum.
'''