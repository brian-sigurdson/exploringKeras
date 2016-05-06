import numpy as np

def load_data_517_1_1_29():
	
	# print("loading data...")
	
	# data_file = "./../data/ff_x_normalized_data.csv"
	data_file = "./../data/ff_x_normalized_data.csv"
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
	x = x.reshape(517,1,1,29)
	print(x.shape)
	
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
	'zeros2 = np.zeros((517,28,29), dtype=np.int)'
	# print("zeros2.shape = ", zeros2.shape)

	# print("x = np.append(x, zeros2, axis=1)")
	'x = np.append(x, zeros2, axis=1)'
	# print("x.shape = ", x.shape)
	# at this point x= 517x29x29

	# pad seems to pad to all axes, so can I pad 3 rows of zeros to all
	# axes and then delete them from the two axes i don't want added?
	# print("x = np.pad(x, (0, toPad), mode='constant', constant_values=(0, 0))")
	
	' x = np.pad(x, (0, 3), mode="constant", constant_values=(0, 0))'
	
	# print("x.shape = ", x.shape)

	# print("np.delete(x, [517, 518, 519], 0) deletes the row at position 520 along the z-axis?")
	
	' x = np.delete(x, [517, 518, 519], 0)'
	
	# print("x = np.delete(x, [517, 518, 519], 0)")
	# print("x.shape = ", x.shape)
	# print(x)
	
	return x

load_data_517_1_1_29()
