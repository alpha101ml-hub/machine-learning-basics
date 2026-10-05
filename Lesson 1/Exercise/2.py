# Create a column vector c of shape (5,1) filled with random integers between 0 and 10

import numpy as np
c = np.random.randint(0, 11, size=(5,1))

print("Column vector c:")
print(c)
print("Shape:", c.shape)
