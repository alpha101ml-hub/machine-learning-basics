A = np.ones((3,4))
B = np.ones((2,1))  # shape (2,1)
# A + B -> ValueError: operands could not be togehter with shapes (3,4) (2,1)

'''
Trailing: 4 and 1 OK. Next: 3 and 2 -> neither 1 nor equal -> error
'''
