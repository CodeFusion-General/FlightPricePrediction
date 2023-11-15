import numpy as np
import pandas as pd

# Step 1: Creating the initial direct-relation matrix Z
Z_data = {
    'Bunker oils': [0, 0.298, 0.185, 0.271, 0.148, 0.174],
    'Ballast waters': [0.04, 0, 0.271, 0.184, 0.204, 0.16],
    'Garbage and Solid Wastes': [0.321, 0.174, 0, 0.316, 0.194, 0.217],
    'Sewages waters': [0.302, 0.185, 0.256, 0.02, 0.301, 0.122],
    'Anti-fouling paints': [0.412, 0.294, 0.112, 0.307, 0, 0.318],
    'Deck, hold, vs. washing operations (Daily opr.)': [0.201, 0.173, 0.304, 0.272, 0.207, 0]
}

Z = pd.DataFrame(Z_data, index=Z_data.keys())

# Display the initial direct-relation matrix Z
print("Table 1. Initial direct matrix Z.")
print(Z)
print("\n")

# Step 2: Calculating the average matrix Z
L = len(Z.columns)
Z_avg = Z.mean(axis=1)

# Step 3: Normalizing the direct-relation matrix
M = Z / L

# Display the normalized direct-relation matrix M
print("Normalized direct-relation matrix M:")
print(M)
print("\n")

# Step 4: Calculating the total-relation matrix K
H = np.identity(L)
K = M.dot(np.linalg.inv(H - M))

# Display the total influential relation matrix K
print("Table 2. Total influential relation matrix K.")
print(K)
print("\n")

# Step 5: Calculating the sum of rows and columns
D_sum = np.sum(K, axis=1)
R_sum = np.sum(K, axis=0)

# Display the sum of influences given and received on each criterion
print("Table 3. Sum of influences given and received on each criterion")
d_r_table = pd.DataFrame({'D + R': D_sum, 'D - R': R_sum})
print(d_r_table)
