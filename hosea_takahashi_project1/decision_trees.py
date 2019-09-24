import numpy as np


'''
Finds the probability of selecting x_value from the array X
X is a np.array
x_value is an integer
'''
def probability(X, x_value):

	if len(X) == 0:
		return 0
	#returns number of elemnts in X that equal x_value divided by the number of elements
	return (np.count_nonzero(X == x_value)) / len(X)

'''
Calculates the entropy of Y
Y: 2D np.array
'''
def entropy(Y):

	#Find the probability that the sample is on the left side of the split (that Y == 0)
	left_probability = probability(Y, 0)

	#Find the probability that the sample is on the left side of the split (that Y == 1)
	right_probability = probability(Y, 1)

	'''Checks if the probability of the left side is zero and if it is assigns left_side to be 0.
	Cannot find log base 2 of 0'''
	if(left_probability == 0):

		left_side = 0
	else:
		#Finds the value for the left side of entropy function
		left_side = -left_probability * np.log2(left_probability)

	'''Checks if the probability of the right side is zero and if it is assigns right_side to be 0.
	Cannot find log base 2 of 0'''
	if(right_probability == 0):

		right_side = 0
	else:
		#Finds the value for the left side of entropy function
		right_side = -right_probability * np.log2(right_probability)

	#Adds the left and right side to calculate the total entropy
	total = left_side + right_side

	return total


'''
Calculates the information gain of the feature being split on feature_index
features:  2D np.array of training data
feature_idex: interger, the index of the feature you want to split on in features
label: 2D np.array of labels
'''
def information_gain(features, feature_index, label):

	#Finds the entropy of the parent labels
	parent_entropy = entropy(label)

	#Finds the entropy of the left side of the split (where features == 0)
	left_entropy = entropy(label[features[:,feature_index] == 0])

	#Finds the entropy of the right side of the split (where features == 1)
	right_entropy = entropy(label[features[:,feature_index] == 1])

	#returns total information gain
	return parent_entropy - (probability(features[:,feature_index], 0) * left_entropy + probability(features[:,feature_index], 1) * right_entropy)


'''
Finds the feature with the highest information gain to split the training data on
features: 2D np.array of training data
label: 2d np.array of labels
'''
def find_best_split(features,label):

	best_index = 0

	best_information_gain = 0

	#Loops through the columns of the training data
	for index in range(np.size(features,1)):

		new_information_gain = information_gain(features, index, label)

		if new_information_gain > best_information_gain :

			best_index = index
			best_information_gain = new_information_gain

	#returns the index of the feature with the best information gain
	return best_index


'''
Recursively builds a tree and returns a list of size three: [feature_split_index, left_subtree, right_subtree]
X: 2D np.array of training data
Y: 2D np.array of labels
max_depth: integer, the maximum depth of the tree
'''
def DT_train_binary(X,Y,max_depth):

	#if the maximum depth is not reached and there are more features to split on
	if not max_depth == 0 and not X.size == 0:

		#finds the index of the best feature to split on and stores it in best_split_index
		best_split_index = find_best_split(X, Y)


		#Splits the training data for the left subtree where the selected feature is 0
		left_training_data = X[X[:,best_split_index] == 0]
		#Splits the labels for the left subtree where the selected feature is 0
		left_labels = Y[X[:,best_split_index] == 0]



		#Splits the training data for the right subtree where the selected feature is 1
		right_training_data = X[X[:,best_split_index] == 1]
		#Splits the labels for the right subtree where the selected feature is 1
		right_labels = Y[X[:,best_split_index] == 1]



		#deletes the column used for the split to pass into left subtree
		left_training_data = np.delete(left_training_data, best_split_index, 1)
		#deletes the column used for the split to pass into right subtree
		right_training_data = np.delete(right_training_data, best_split_index, 1)



		#if the entropy is 0 then there is no need to continue splitting and we can make a decision
		if not entropy(left_labels) == 0:

			#recursively calls DT_train_binary on the left side of the split
			left_subtree = DT_train_binary(left_training_data, left_labels, max_depth - 1)
		
		else:

			#Makes a decision with the first element of left_labels
			left_subtree = left_labels[0]


		#if the entropy is 0 then there is no need to continue splitting and we can make a decision
		if not entropy(right_labels) == 0:	
			#recursively calls DT_train_binary on the right side of the split
			right_subtree = DT_train_binary(right_training_data, right_labels, max_depth - 1)

		else:
			#Makes a decision with the first element of right_labels
			right_subtree = right_labels[0]



		'''
		store the tree as a list of three, with the first element being the index of the feature split,
		the second being the left subtree, and the third element being the right subtree
		'''
		return [best_split_index,left_subtree, right_subtree]


	#if the max depth is 0 or there are no features left to split on, we will return the label 
	else:
		
		#find the probability that the label is 0
		probability_0 = probability(Y, 0)
		#find the probability that the label is 1
		probability_1 = probability(Y, 1)

		#return the decision for the label with the highest probability of being correct
		if probability_0 > probability_1 :

			return 0

		else :

			return 1


def DT_test_binary(X,Y,DT):

	#List that predicts the label of the sample
	DT_prediction = []

	#Loops through the sample data to get predictions
	for sample in X:

		#Reset DT_tree to the beginning of the tree, DT
		DT_tree = DT

		#While the length of the tree list is 3 then there is no decision to be made and we will continue down the tree
		while len(DT_tree) == 3:

			#if the feature is 0, go to the left subtree
			if sample[DT_tree[0]] == 0:

				DT_tree = DT_tree[1]

			#else go to the right subtree
			else:

				DT_tree = DT_tree[2]

		#Append the prediction to the DT_prediction list
		DT_prediction.append(DT_tree)

	#Find the accuracy by counting the number of correct predictions divided by the total number of samples
	count_correct = 0

	for index in range(len(Y)):

		if DT_prediction[index] == Y[index]:

			count_correct = count_correct +1

	return count_correct/len(Y)


def DT_train_binary_best(X_train, Y_train, X_val, Y_val):

	new_accuracy = 0
	old_accuracy = -1
	depth = 0

	DT_new = DT_train_binary(X_train,Y_train, depth)

	while new_accuracy > old_accuracy:

		depth = depth + 1

		DT_old = DT_new
		DT_new = DT_train_binary(X_train,Y_train, depth)

		old_accuracy = new_accuracy
		new_accuracy = DT_test_binary(X_val,Y_val, DT_new)


	return DT_old

def DT_make_prediction(x,DT):

	while len(DT) == 3:

		#if the feature is 0, go to the left subtree
		if x[DT[0]] == 0:

			DT = DT[1]

		#else go to the right subtree
		else:

			DT = DT[2]

	#Return the prediction
	return DT








'''
Finds the probability of selecting x_value from the array X
X is a np.array
x_value is an integer
'''
def probability_real(X, x_value, is_less_than = True):

	if len(X) == 0:
		return 0
	
	if is_less_than == True:
		#returns number of elemnts in X that are less than x_value divided by the number of elements
		return (np.count_nonzero(X < x_value)) / len(X)

	else:
		return (np.count_nonzero(X >= x_value)) / len(X)


'''
Calculates the information gain of the feature being split on feature_index
features:  2D np.array of training data
feature_idex: interger, the index of the feature you want to split on in features
label: 2D np.array of labels
'''
def information_gain_real(features, feature_index, label, split_value):

	#Finds the entropy of the parent labels
	parent_entropy = entropy(label)

	#Finds the entropy of the left side of the split (where features < split_value)
	left_entropy = entropy(label[features[:,feature_index] < split_value])

	#Finds the entropy of the right side of the split (where features >= split_value)
	right_entropy = entropy(label[features[:,feature_index] >= split_value])

	#returns total information gain
	return parent_entropy - (probability_real(features[:,feature_index], split_value, True) * left_entropy + probability_real(features[:,feature_index], split_value, False) * right_entropy)


'''
Finds the feature with the highest information gain to split the training data on
features: 2D np.array of training data
label: 2d np.array of labels
'''
def find_best_split_real(features,label):

	best_index = 0
	split_value = 0

	best_information_gain = 0

	for index in range(np.size(features,1)):

		for value in features[:,index]:

			new_information_gain = information_gain_real(features, index, label, value)

			if new_information_gain > best_information_gain :

				best_index = index
				split_value = value
				best_information_gain = new_information_gain

	return [best_index, split_value]

'''
Recursively builds a tree and returns a list of size three: [feature_split_index, left_subtree, right_subtree]
X: 2D np.array of training data
Y: 2D np.array of labels
max_depth: integer, the maximum depth of the tree
'''
def DT_train_real(X,Y,max_depth):

	#if the maximum depth is not reached and there are more features to split on
	if not max_depth == 0 and not X.size == 0:

		#finds the index of the best feature to split on and stores it in best_split_index
		best_split_index, best_split_value = find_best_split_real(X, Y)

		

		#Splits the training data for the left subtree where the selected feature is < best_split_value
		left_training_data = X[X[:,best_split_index] < best_split_value]
		#Splits the labels for the left subtree where the selected feature is < best_split_value
		left_labels = Y[X[:,best_split_index] < best_split_value]



		#Splits the training data for the right subtree where the selected feature is >= best_split_value
		right_training_data = X[X[:,best_split_index] >= best_split_value]
		#Splits the labels for the right subtree where the selected feature is >= best_split_value
		right_labels = Y[X[:,best_split_index] >= best_split_value]



		#deletes the column used for the split to pass into left subtree
		#left_training_data = np.delete(left_training_data, best_split_index, 1)
		#deletes the column used for the split to pass into right subtree
		#right_training_data = np.delete(right_training_data, best_split_index, 1)



		#if the entropy is 0 then there is no need to continue splitting and we can make a decision
		if not entropy(left_labels) == 0:

			#recursively calls DT_train_binary on the left side of the split
			left_subtree = DT_train_binary(left_training_data, left_labels, max_depth - 1)
		
		else:

			#Makes a decision with the first element of left_labels
			left_subtree = left_labels[0]


		#if the entropy is 0 then there is no need to continue splitting and we can make a decision
		if not entropy(right_labels) == 0:	
			#recursively calls DT_train_binary on the right side of the split
			right_subtree = DT_train_binary(right_training_data, right_labels, max_depth - 1)

		else:
			#Makes a decision with the first element of right_labels
			right_subtree = right_labels[0]



		'''
		store the tree as a list of three, with the first element being the index of the feature split,
		the second being the left subtree, and the third element being the right subtree
		'''
		return [(best_split_index,best_split_value),left_subtree, right_subtree]


	#if the max depth is 0 or there are no features left to split on, we will return the label 
	else:
		
		#find the probability that the label is 0
		probability_0 = probability(Y, 0)
		#find the probability that the label is 1
		probability_1 = probability(Y, 1)

		#return the decision for the label with the highest probability of being correct
		if probability_0 > probability_1 :

			return 0

		else :

			return 1

def DT_test_real(X,Y,DT):

	#List that predicts the label of the sample
	DT_prediction = []

	#Loops through the sample data to get predictions
	for sample in X:

		#Reset DT_tree to the beginning of the tree, DT
		DT_tree = DT

		#While the length of the tree list is 3 then there is no decision to be made and we will continue down the tree
		while len(DT_tree) == 3:

			#if the feature is 0, go to the left subtree
			if sample[DT_tree[0][0]] < DT_tree[0][1]:

				DT_tree = DT_tree[1]

			#else go to the right subtree
			else:

				DT_tree = DT_tree[2]

		#Append the prediction to the DT_prediction list
		DT_prediction.append(DT_tree)

	#Find the accuracy by counting the number of correct predictions divided by the total number of samples
	count_correct = 0

	for index in range(len(Y)):

		if DT_prediction[index] == Y[index]:

			count_correct = count_correct +1

	return count_correct/len(Y)


def DT_train_real_best(X_train,Y_train,X_val,Y_val):

	new_accuracy = 0
	old_accuracy = -1
	depth = 0

	DT_new = DT_train_real(X_train,Y_train, depth)

	while new_accuracy > old_accuracy:

		depth = depth + 1

		DT_old = DT_new
		DT_new = DT_train_real(X_train,Y_train, depth)

		old_accuracy = new_accuracy
		new_accuracy = DT_test_real(X_val,Y_val, DT_new)


	return DT_old

def DT_make_prediction_real(x,DT):

	#While the length of the tree list is 3 then there is no decision to be made and we will continue down the tree
	while len(DT) == 3:

		#if the feature is 0, go to the left subtree
		if x[DT[0][0]] < DT[0][1]:

			DT = DT[1]

		#else go to the right subtree
		else:

			DT = DT[2]

	#Append the prediction to the DT_prediction list
	return DT


