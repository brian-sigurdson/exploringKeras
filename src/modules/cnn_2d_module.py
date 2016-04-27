import numpy as np

'''
currently a single function to 
1) retrieve the forest fire data set
2) pad it
3) reshape it
so that it is in the format expected by the example program mnist_cunn.py
'''

def load_data():
	# all the values are hard coded for now
	# not generalizing at the moment
	
	# print("loading data...")
	x = np.loadtxt("./../data/ff_x_data.csv", delimiter=',')
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
	# print("x.shape = ", x.shape)
	# print(x)
	
	return x
