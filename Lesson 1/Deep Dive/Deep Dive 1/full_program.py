# Add a column vector to every column of a matrix
col_vec = np.array(([1],[2],[3]))  # shape (3,1)
matrix = np.array(([10,20,30],
                   [40,50,60],
                   [70,80,90]))  # shape (3,3)
print(matrix + col_vec)
print(result)
# col_vec (3,1) broadcasts to (3,3) by repeating horizontally
# Output:
# [[11 21 31]]
# [[42 52 62]]
# [[73 83 93]]

'''
Broadcasting never moves data in memory - it's a stride trick. That's why it's fast/
'''
