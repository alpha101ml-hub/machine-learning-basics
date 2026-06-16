import numpy as np

A = np.ones((3,4))  # shape (3,4)
B = np.ones((3,1))  # shape (3,1)
c = A + B

'''
Trainling dimensions: 4 and 1 -> 1 is compatible (broadcast across columns).
Next: 3 and 3 -> equal.
'''


