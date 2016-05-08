########################################################
# the following is fine for this small comparison network, 
# but when you go to implement more extensive testing, then 
# make use of objects as function parameters and not simply numbers

# they you don't need to hunt around to make sure all of the parameter
# lists are updated for each minor change.


import modules.ff_mlp_module as ffmlp

# print file name
print("running ff_mlp_classify_v_001.py")

#######################################################################
# define some variables to reuse
#######################################################################
epochs = 3000
batch_size = 32
print("epochs = ", epochs)

# specify label file
num_labels = 3
labels_file = "../data/ff_" + str(num_labels) + "_labels.csv"

# specify data file
# data_file = "./../data/ff_x_zscored_data.csv"
# data_file = "./../data/ff_x_normalized_data.csv"
# data_file = "./../data/ff_x_normalized_data_2.csv"
data_file = "./../data/ff_x_normalized_data_1.csv"

hidden_activation = 'relu'
# hidden_activation = 'tanh'
# hidden_activation = 'sigmoid'

# i think i need to do more than just change the output layer and loss function to 
# change from classification to regression problem
output_activation = 'softmax'
# output_activation = 'relu'

# for a multi-class classification problem
loss_func = 'categorical_crossentropy'
# for a mean squared error regression problem
# loss_func = 'mse'
              

# to avoid having to reload the data multiple times you can store it
data, nb_classes = ffmlp.load_data(labels_file, data_file)
# NOTE:  I think you need to manually override nb_classes = 1 at this point,
# if you're doing regression, in addition to changing the activation and or output function
# nb_classes = 1

#######################################################################

'''
# 1: Two layer networks - many
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=200, layer2=100, epochs=epochs, 
	batch=batch_size)

reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=100, layer2=50, epochs=epochs, 
	batch=batch_size)
	
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=50, layer2=25, epochs=epochs, 
	batch=batch_size)
'''
'''
# 2: Three layer networks - many
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=200, layer2=100, layer3=12, 
	epochs=epochs, batch=batch_size)

reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=100, layer2=50, layer3=12, 
	epochs=epochs, batch=batch_size)
	
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=50, layer2=25, layer3=12, 
	epochs=epochs, batch=batch_size)
'''
'''
# 3: Two layer networks - few
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=29, layer2=15, epochs=epochs, 
	batch=batch_size)

reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=25, layer2=15, epochs=epochs, 
	batch=batch_size)
	
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=20, layer2=15, epochs=epochs, 
	batch=batch_size)
'''
'''
# 4: Three layer networks - few
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=29, layer2=15, layer3=12, 
	epochs=epochs, batch=batch_size)

reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=25, layer2=15, layer3=12, 
	epochs=epochs, batch=batch_size)
	
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=20, layer2=15, layer3=12, 
	epochs=epochs, batch=batch_size)
'''

# 5: Four layer networks - few 
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=40, layer2=30, layer3=20, layer4=15,
	epochs=epochs, batch=batch_size)
'''
# 6: Four layer networks - many
reload(ffmlp)	
model, losses = ffmlp.run_network(
	labels_file, data_file, hidden_activation, output_activation, loss_func, data=data, 
	nb_classes=nb_classes, layer1=250, layer2=125, layer3=60, layer4=30,
	epochs=epochs, batch=batch_size)
'''
