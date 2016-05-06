from __future__ import print_function 
'demanded to be the first line: from __future__ import print_function '

'''
CS5850 - Spring 2016
Brian Sigurdson

This is a first attempt / test run at:
1) loading the forest fire data into numpy arrays
2) use scikit-learn to select training and test data
3) modify keras CNN example to attemt to train a CNN on ff data
4) if successful, then possibly follow-up with over/under sampling of small data sets using the forest fire data

'''

# imports
from sklearn.cross_validation import StratifiedKFold
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils


'function to load forest fire data'
import modules.cnn_2d_module as cnn2d

# print file name
print("running cnn_2d_classify_v_004.py")

######################################################################
# define some variables to reuse
######################################################################
nb_epoch = 300
batch_size = 32

# load lable data directly from file
num_labels = 2
labels_file = "../data/ff_" + str(num_labels) + "_labels.csv"
y, nb_classes, nb_folds = cnn2d.load_y_data(labels_file)

# specify data file
# data_file = "./../data/ff_x_zscored_data.csv"
data_file = "./../data/ff_x_normalized_data.csv"

hidden_activation = 'relu'
output_activation = 'softmax'
# for a multi-class classification problem
loss_func = 'categorical_crossentropy'
# for a mean squared error regression problem
# loss_func = 'mse'

# input image dimensions
img_rows = 32
img_cols = 32

# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_col = 3
nb_row = 3
######################################################################


# 1) loading data via a module to facilitate some reshaping of the data
x = cnn2d.load_data_517_1_32_32(data_file)

# 2) split it with scikit-learn
skf = StratifiedKFold(y, n_folds=nb_folds)


for train_index, test_index in skf:
	X_train, X_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


model = Sequential()

# follow as close as I can to the example in 10.5 of Heaton's AIFH v3 pg 199
# I just modified the Heaton's example from pg. 205, b/c the illustration
# on pg. 199 keeps throwing error.  I can't figure out right now.

# layer
model.add(Convolution2D(32, 3, 3,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation(hidden_activation))

# layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# layer
model.add(Convolution2D(64, 2, 2))
model.add(Activation(hidden_activation))

# layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# layer
model.add(Convolution2D(128, 2, 2))
model.add(Activation(hidden_activation))

# layer
model.add(MaxPooling2D(pool_size=(2, 2)))


# layer (i added these two layers
model.add(Convolution2D(256, 2, 2))
model.add(Activation(hidden_activation))
# layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# layer
# added this layer due to exception being thrown without it
# not part of heaton's example
model.add(Flatten())

# layer
model.add(Dense(500))
model.add(Activation(hidden_activation))

# layer
model.add(Dense(nb_classes))
model.add(Activation(output_activation))

model.compile(loss=loss_func, optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=0, validation_data=(X_test, Y_test))
          
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])

