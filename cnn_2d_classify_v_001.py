from __future__ import print_function 
'demanded to be the first line: from __future__ import print_function '

'''
CS5850 - Spring 2016
Brian Sigurdson

This is a first attempt / test run at:
1) loading the ff data into numpy arrays
2) use scikit-learn to select training and test data
3) modify keras CNN example to attemt to train a CNN on ff data
4) if successful, then possibly follow-up with over/under sampling of small data sets using ff data

*** NOTE ****
As soon as I get a working model, then start to version my code and 
push it to the repo
'''

'imports'
from sklearn.cross_validation import StratifiedKFold
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

'function to load forest fire data'
import modules.cnn_2d_module as ffmod

'parameters'
'''rows = 517
cols = 30'''


'1) read in the ff data, it is in the same dirctory for now '
print('Loading data...')

'X = np.zeros((rows, cols))'
'read the data into the array'
'x = np.loadtxt("ff_x_data.csv", delimiter=',')'

x = ffmod.load_data()
print("x.dim= ", x.ndim, "x.shape=", x.shape)


'i thought i might need to transpose the Y array, but the transpose is the same as the original'
'y = np.loadtxt("ff_y_data.csv", delimiter=',')'
y = np.loadtxt("data/ff_labels.csv", dtype=int, delimiter=',')

print("y.dim= ", y.ndim, "y.shape=", y.shape)
'print(x)'
'print(y)'


'2) split it with scikit-learn'
skf = StratifiedKFold(y, n_folds=2)
print('len(skf) = ', len(skf))

'print("skf = ", skf)'

for train_index, test_index in skf:
	X_train, X_test = x[train_index], x[test_index]
	y_train, y_test = y[train_index], y[test_index]
	

print("X_train.dim = ", X_train.ndim)
print("X_train.shape =", X_train.shape)

print("X_test.dim = ", X_test.ndim)
print("X_test.shape =", X_test.shape)

print("y_train.dim = ", y_train.ndim)
print("y_train.shape =", y_train.shape)

print("y_test.dim = ", y_test.ndim)
print("y_test.shape =", y_test.shape)

'''
print("Train index:", train_index)
print('len(train_index', len(train_index))
print("Test index:", test_index)
print('len(test_index', len(test_index))
print("x_train", x_train)
print('len(x_train', len(x_train))
print("x_test", x_test)
print('len(x_test', len(x_test))
print("y_train", y_train)
print('len(y_train', len(y_train))
print("y_test", y_test)
print('len(y_test', len(y_test))
'''	

'3) now tran a network '
'using conv2D https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py'

''' (following note is from original file)
Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

'''
2016-04-17:  
I will basically modify as few parameters as possible to try and make
the forest fire data "look" like an image to the network.

I may need to add an extra row or column for padding, if the model
cannot act on vectors.

I started with the 1D example, but many of the parameters were unfamiliar
and reading the documentation didn't provide clairification on their use.

I should have started with the 2D model as the model itself and the
paramaters seems to be a bit more familiar.  Probably because image
analysis is a common example with CNN's and the Heaton text gave a
nice discussion of CNN's in the context of image analysis.

Also, the conversation here
https://groups.google.com/forum/#!topic/keras-users/ALfpSnvZkVk
suggested that 2D was a better approach.

'''


'''
# original parameters
batch_size = 128
nb_classes = 10
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
'''

# new parameters
batch_size = 128
''' 
i believe the classes is the for the final number of
output nodes.  i have 7 labels, so try 7
'''
' eight classes = 0..7'
nb_classes = 8
'nb_epoch = 12'
nb_epoch = 12

# input image dimensions
img_rows = 32
img_cols = 32

# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3


X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

''' ================================================================ '''
'''
Traceback (most recent call last):
  File "attempt_2D_CNN.py", line 185, in <module>
    Y_train = np_utils.to_categorical(y_train, nb_classes)
  File "build/bdist.linux-x86_64/egg/keras/utils/np_utils.py", line 17, in to_categorical
IndexError: index 7 is out of bounds for axis 1 with size 7

the error seems to be coming from the first of the next couple lines

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

'''


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)



model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

