import numpy as np

def load_y_data(labels_file):
	''' load the labels / truth values for the forest fire data set '''
	
	print("using labels_file: ", labels_file)
	y = np.loadtxt(labels_file, dtype=int, delimiter=',')
	
	# determine the number of labels using a set object
	yset = set(y)
	n_classes = len(yset)
	
	# dictionary to hold class counts
	class_cnts = {}
	
	# determine the size of the classes in y
	# for each element in the set
	for cnt1 in yset:
		# cound the number of times it is in y
		for cnt2 in range(0, len(y)):
			# if we find the key is already present
			if(class_cnts.has_key(cnt1)):
				# increment the key
				val = class_cnts.get(cnt1) + 1
				class_cnts[cnt1] = val
			else:
				# start a new key
				class_cnts[cnt1] = 1
	
	# which class has the largest number of elements
	min_folds = 600
	for cnt1 in yset:
		if(class_cnts[cnt1] < min_folds):
			min_folds = class_cnts[cnt1]
	
	# try to set to 5 cross-fold if possible, else default=3, else 2 if necessary
	# let it throw error if min_folds=1, but shouldn't happen since
	# i'm determining labels
	if(min_folds > 5):
		min_folds = 5
	elif(min_folds > 2):
		min_folds = 3
	else:
		min_folds = 2

	print("min_folds = ", min_folds)
	
	return [y, n_classes, min_folds]
	

'''
currently a single function to 
1) retrieve the forest fire data set
2) pad it
3) reshape it
so that it is in the format expected by the example program mnist_cunn.py
'''

def load_data_517_1_32_32(data_file):
	'''
	Read the forest fire data set from a file and reshape it to make it
	similar in size to the example 2D CNN network being used as a guide.
	'''
	# all the values are hard coded for now
	# not generalizing at the moment
	
	# print("loading data...")
	
	# data_file = "./../data/ff_x_zscored_data.csv"
	# data_file = "./../data/ff_x_normalized_data.csv"
	x = np.loadtxt(data_file, delimiter=',')
	print("using data_file: ", data_file)
	
	# print("x.dim= ", x.ndim, "x.shape=", x.shape)
	# print("len(x.shape)=", len(x.shape))
	# print("x.shape = ", x.shape)
	# print("x = ")
	# print(x)

	# print("try reshaping (517,29) to (517,1,29")
	# print("x = x.reshape(517,1,29)")
	x = x.reshape(517,1,29)
	# print("x.dim= ", x.ndim, "x.shape=", x.shape)
	# print("len(x.shape)=", len(x.shape))
	# print("x.shape = ", x.shape)
	# print("x = ")
	# print(x)

	# print("x.shape = ", x.shape)
	# print("i have a 517x1x29 matrix, can append a 517x28x29 matrix")
	# print("to get a 517x29x29 matrix?")

	# i might keep it at 517x1x29x29
	# if i don't find a simple way to get the matrix to 517x32x32
	# print("zeros2 = np.zeros((517,28,29), dtype=np.int)")
	zeros2 = np.zeros((517,28,29), dtype=np.int)
	# print("zeros2.shape = ", zeros2.shape)

	# print("x = np.append(x, zeros2, axis=1)")
	x = np.append(x, zeros2, axis=1)
	# print("x.shape = ", x.shape)
	# at this point x= 517x29x29

	# pad seems to pad to all axes, so can I pad 3 rows of zeros to all
	# axes and then delete them from the two axes i don't want added?
	# print("x = np.pad(x, (0, toPad), mode='constant', constant_values=(0, 0))")
	x = np.pad(x, (0, 3), mode='constant', constant_values=(0, 0))
	# print("x.shape = ", x.shape)

	# print("np.delete(x, [517, 518, 519], 0) deletes the row at position 520 along the z-axis?")
	x = np.delete(x, [517, 518, 519], 0)
	# print("x = np.delete(x, [517, 518, 519], 0)")
	print("x.shape = ", x.shape)
	# print(x)
	
	return x
	
	
def load_data_517_1_1_29():
	
	data_file = "./../data/ff_x_normalized_data.csv"
	x = np.loadtxt(data_file, delimiter=',')
	print("using data_file: ", data_file)
	
	x = x.reshape(517,1,29)
	x = x.reshape(517,1,1,29)
	# print(x.shape)
		
	return x
	
