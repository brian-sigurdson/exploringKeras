from sklearn.cross_validation import StratifiedKFold
import numpy as np
np.random.seed(1337)  # for reproducibility

import time
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist


def load_y_data(labels_file):
	''' load the labels / truth values for the forest fire data set '''
	
	print("using labels_file: ", labels_file)
	# y = np.loadtxt("../data/ff_2_labels.csv", dtype=int, delimiter=',')
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
	
	
def load_data_517_1_32_32(data_file):
	'''
	Read the forest fire data set from a file and reshape it to make it
	similar in size to the example 2D CNN network being used as a guide.
	
	This function returns a numpy array (517x29) of forest fire data
	'''
	
	
	# data_file = "./../data/ff_x_zscored_data.csv"
	# data_file = "./../data/ff_x_normalized_data.csv"
	x = np.loadtxt(data_file, delimiter=',')
	print("using data_file: ", data_file)
	
	
	'''
	** I think I may well be able to simply return the data straight
	from the forest fire csv file.
	
	** I belive that the keras feedforward tutorial,	
	https://github.com/Vict0rSch/deep_learning/tree/master/keras/feedforward
	and example,
	https://github.com/Vict0rSch/deep_learning/blob/master/keras/feedforward/feedforward_keras_mnist.py
	transform the image data into a numpy 60,000 x 784 array, where
	784 = 28 * 28 = rows * columns.
	
	** think i can use the feedforward code for the y-values also.
	** actually, it is the same as alredy in my current code.
	
	'''
	
	return x

##############################################################
# below here is the code for the Keras tutorial that I'll use as a 
# basis for my comparitive feedforward mlp

'''
keras tutorial found here:
https://github.com/Vict0rSch/deep_learning/tree/master/keras/feedforward
'''

######## class LossHistory (call back) #######################
class LossHistory(cb.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		batch_loss = logs.get('loss')
		self.losses.append(batch_loss)
######## class LossHistory (call back) #######################


def load_data(labels_file, data_file):
	''' split and return the training and test data'''
	
	print("Loading data...")
	# currently no reshaping going on for load_x_data()
	# returns a 517x29 numpy array
	x = load_data_517_1_32_32(data_file)
	# load_y_data() returns a 517x1 numpy array
	y, nb_classes, num_folds = load_y_data(labels_file)
	
	# split it with scikit-learn
	skf = StratifiedKFold(y, n_folds=num_folds)
	
	for train_index, test_index in skf:
		X_train, X_test = x[train_index], x[test_index]
		y_train, y_test = y[train_index], y[test_index]
	
	# convert class vectors to binary class matrices
	y_train = np_utils.to_categorical(y_train, nb_classes)
	y_test = np_utils.to_categorical(y_test, nb_classes)
		
	# they were reshaping to make an image look like a big matrix
	# I don't think i need to do any reshaping here, b/c my x_test
	# and x_train s/b Ax29 and Bx29 where A+B=517
    # X_train = np.reshape(X_train, (60000, 784))
    # X_test = np.reshape(X_test, (10000, 784))		

	print ("Data loaded.")	
	return [X_train, X_test, y_train, y_test], nb_classes
	

# I don't think there is much to change here.  I'll leave a lot of 
# defaults for now, except input size.
def init_model(nb_classes, activation, output_activation, loss_func, layer1, layer2=None, layer3=None, layer4=None):
	
	print('Compiling Model ... ')
	start_time = time.time()
	
	# declare model
	model = Sequential()
	
	# layer 1
	#model.add(Dense(500, input_dim=784))
	model.add(Dense(layer1, input_dim=29))
	model.add(Activation(activation))
	model.add(Dropout(0.4))

	if(layer2!=None):
		# include layer 2
		# model.add(Dense(300))
		model.add(Dense(layer2))
		model.add(Activation(activation))
		model.add(Dropout(0.4))

	if(layer3!=None):
		# include layer 3
		model.add(Dense(layer3))
		model.add(Activation(activation))
		model.add(Dropout(0.4))

	if(layer4!=None):
		# include layer 4
		model.add(Dense(layer4))
		model.add(Activation(activation))
		model.add(Dropout(0.4))
		
    # output layer
    # indicate the number of outputs
	model.add(Dense(nb_classes))
	model.add(Activation(output_activation))

	rms = RMSprop()
    # model.compile(loss='categorical_crossentropy', optimizer=rms)
	model.compile(loss=loss_func, optimizer=rms, metrics=['accuracy'])
    
	print ('Model compield in {0} seconds'.format(time.time() - start_time))
    
	return model
    

def run_network(labels_file, data_file, activation, output_activation, loss_func, nb_classes, layer1, layer2=None, layer3=None, layer4=None,
		data=None, model=None, epochs=20, batch=256):

    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test, nb_classes = load_data(labels_file, data_file)
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model(nb_classes, activation, output_activation, loss_func, layer1, layer2, layer3, layer4)

        history = LossHistory()

        print 'Training model...'

        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_test), verbose=0)

        print "Training duration : {0}".format(time.time() - start_time)

        # score = model.evaluate(X_test, y_test, batch_size=16)
        score = model.evaluate(X_test, y_test, batch_size=batch)
        
        # print("score = model.evaluate: ", score)
        print "Network's test score [loss, accuracy]: {0}".format(score)
        
        return model, history.losses
    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses


def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()
