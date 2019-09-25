import numpy as np
import decision_trees as dt

# Default max depth value
max_depth = 2
# Training Set 1
X_1 = np.array([[0, 1], [0, 0], [1, 0], [0, 0],  [1, 1]])
Y_1 = np.array([[1], [0], [0], [0], [1]])
# Validation Set 1
X_val_1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_val_1 = np.array([[0], [1], [0], [1]])
# Test Set 1
X_test_1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y_test_1 = np.array([[1], [1], [0], [1]])



# Training Set 2
X_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
Y_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
# Validation Set 2
X_val_2 = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
Y_val_2 = np.array([[0], [0], [1], [0], [1], [1]])
# Test Set 2
X_test_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
Y_test_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])

tree = dt.DT_train_binary(X_2,Y_2,-1)
tree_1 = dt.DT_train_binary_best(X_2,Y_2,X_val_2,Y_val_2)

print(dt.DT_test_binary(X_test_2,Y_test_2,tree))
print(dt.DT_test_binary(X_test_2,Y_test_2,tree_1))

print(tree_1)

# Inviting friends over for dinner problem
X_invite_friends = np.array([[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]])
Y_invite_friends = np.array([[1], [1], [0], [1], [0], [1], [1], [0]])
# max_depth = 5  # Custom max depth value for the friends problem

# Make Prediction function data set
X_prediction = np.array([[0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]])
Y_prediction = np.array([[0], [1], [0]])

# Real value data set
X_real = np.array([[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2], [5, 3.4, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.7, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1]])
Y_real = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]])