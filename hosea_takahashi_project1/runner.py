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

tree = dt.DT_train_binary(X_1,Y_1,max_depth)
test = dt.DT_test_binary(X_test_1,Y_test_1,tree)

tree_best = dt.DT_train_binary_best(X_1,Y_1,X_val_1,Y_val_1)
test_best = dt.DT_test_binary(X_test_1,Y_test_1,tree_best)


print("Training Set 1:")
print("Binary Test (max_depth = 2): ",test)
print("Binary Test (Best Depth): ",test_best)
print("Decision Tree (max_depth = 2): ",tree)
print("Decision Tree (Best Depth): ",tree_best)


# Training Set 2
X_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
Y_2 = np.array([[0], [1], [0], [0], [1], [0], [1], [1], [1]])
# Validation Set 2
X_val_2 = np.array([[1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
Y_val_2 = np.array([[0], [0], [1], [0], [1], [1]])
# Test Set 2
X_test_2 = np.array([[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 0, 0], [1, 0, 0, 1], [0, 1, 0, 1], [0, 1, 0, 0]])
Y_test_2 = np.array([[1], [1], [0], [0], [1], [0], [1], [1], [1]])


tree = dt.DT_train_binary(X_2,Y_2,max_depth)
test = dt.DT_test_binary(X_test_2,Y_test_2,tree)

tree_best = dt.DT_train_binary_best(X_2,Y_2,X_val_2,Y_val_2)
test_best = dt.DT_test_binary(X_test_2,Y_test_2,tree_best)

print("\nTraining Set 2:")
print("Binary Test (max_depth = 2): ",test)
print("Binary Test (Best Depth): ",test_best)
print("Decision Tree (max_depth = 2): ",tree[0])
print("Decision Tree (Best Depth): ",tree_best)


# Inviting friends over for dinner problem
X_invite_friends = np.array([[1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 1], [0, 1, 0, 0, 1, 0, 0], [1, 1, 0, 1, 0, 0, 1], [1, 0, 1, 0, 1, 1, 1], [1, 1, 0, 1, 1, 0, 1], [1, 1, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1, 1]])
Y_invite_friends = np.array([[1], [1], [0], [1], [0], [1], [1], [0]])
# max_depth = 5  # Custom max depth value for the friends problem

tree_first_5 = dt.DT_train_binary(X_invite_friends[:5],Y_invite_friends[:5],5)


tree_middle_5 = dt.DT_train_binary(X_invite_friends[1:6],Y_invite_friends[1:6],5)


tree_last_5 = dt.DT_train_binary(X_invite_friends[3:],Y_invite_friends[3:],5)


# Make Prediction function data set
X_prediction = np.array([[0, 1, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 0]])
Y_prediction = np.array([[0], [1], [0]])


prediction = []

prediction.append(dt.DT_make_prediction(X_prediction[0],tree_first_5))
prediction.append(dt.DT_make_prediction(X_prediction[0],tree_middle_5))
prediction.append(dt.DT_make_prediction(X_prediction[0],tree_last_5))

print("Sample 1 Prediction: ",prediction,"\n\n")

prediction = []

prediction.append(dt.DT_make_prediction(X_prediction[1],tree_first_5))
prediction.append(dt.DT_make_prediction(X_prediction[1],tree_middle_5))
prediction.append(dt.DT_make_prediction(X_prediction[1],tree_last_5))

print("Sample 2 Prediction: ",prediction,"\n\n")

prediction = []

prediction.append(dt.DT_make_prediction(X_prediction[2],tree_first_5))
prediction.append(dt.DT_make_prediction(X_prediction[2],tree_middle_5))
prediction.append(dt.DT_make_prediction(X_prediction[2],tree_last_5))

print("Sample 3 Prediction: ",prediction,"\n\n")


# Real value data set
X_real = np.array([[4.8, 3.4, 1.9, 0.2], [5, 3, 1.6, 1.2], [5, 3.4, 1.6, 0.2], [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2], [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.4], [7, 3.2, 4.7, 1.4], [6.4, 3.2, 4.7, 1.5], [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4, 1.3], [6.5, 2.8, 4.6, 1.5], [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1]])
Y_real = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [0], [0], [0], [0], [0], [0], [0], [0]])

tree = dt.DT_train_real(X_real,Y_real,-1)


print(tree)


