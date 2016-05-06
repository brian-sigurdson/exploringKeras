########################################################
# code for Keras feedforward mlp tutorial

import modules.feedforward_keras_mnist as fkm

# to avoid having to reload the data multiple times
data = fkm.load_data()
model, losses = fkm.run_network(data=data)

# now if you change some parameters in your code, simply 
'''
reload(fkm)
model, losses = fkm.run_network(data=data)
'''

# code for Keras feedforward mlp tutorial
########################################################

'''
the result of running the tutorial code

Using Theano backend.
Using gpu device 1: GeForce GT 730 (CNMeM is disabled, cuDNN 5004)
DEPRECATION WARNING: softsign was moved from theano.sandbox.softsign to theano.tensor.nnet.nnet 
Loading data...
Data loaded.
Compiling Model ... 
Model compield in 0.695772886276 seconds
Training model...
Train on 60000 samples, validate on 10000 samples
Epoch 1/20
5s - loss: 0.4091 - acc: 0.8744 - val_loss: 0.1750 - val_acc: 0.9441
Epoch 2/20
5s - loss: 0.1764 - acc: 0.9473 - val_loss: 0.1072 - val_acc: 0.9662
Epoch 3/20
5s - loss: 0.1316 - acc: 0.9605 - val_loss: 0.0870 - val_acc: 0.9723
Epoch 4/20
5s - loss: 0.1059 - acc: 0.9682 - val_loss: 0.0810 - val_acc: 0.9751
Epoch 5/20
5s - loss: 0.0915 - acc: 0.9723 - val_loss: 0.0757 - val_acc: 0.9760
Epoch 6/20
5s - loss: 0.0807 - acc: 0.9756 - val_loss: 0.0652 - val_acc: 0.9797
Epoch 7/20
5s - loss: 0.0704 - acc: 0.9784 - val_loss: 0.0670 - val_acc: 0.9784
Epoch 8/20
5s - loss: 0.0647 - acc: 0.9793 - val_loss: 0.0642 - val_acc: 0.9806
Epoch 9/20
5s - loss: 0.0589 - acc: 0.9811 - val_loss: 0.0596 - val_acc: 0.9825
Epoch 10/20
5s - loss: 0.0530 - acc: 0.9834 - val_loss: 0.0562 - val_acc: 0.9830
Epoch 11/20
5s - loss: 0.0504 - acc: 0.9834 - val_loss: 0.0614 - val_acc: 0.9820
Epoch 12/20
5s - loss: 0.0452 - acc: 0.9853 - val_loss: 0.0567 - val_acc: 0.9838
Epoch 13/20
5s - loss: 0.0429 - acc: 0.9863 - val_loss: 0.0608 - val_acc: 0.9831
Epoch 14/20
5s - loss: 0.0401 - acc: 0.9869 - val_loss: 0.0620 - val_acc: 0.9823
Epoch 15/20
5s - loss: 0.0385 - acc: 0.9874 - val_loss: 0.0541 - val_acc: 0.9854
Epoch 16/20
5s - loss: 0.0338 - acc: 0.9891 - val_loss: 0.0605 - val_acc: 0.9841
Epoch 17/20
5s - loss: 0.0324 - acc: 0.9892 - val_loss: 0.0554 - val_acc: 0.9842
Epoch 18/20
5s - loss: 0.0327 - acc: 0.9889 - val_loss: 0.0589 - val_acc: 0.9836
Epoch 19/20
5s - loss: 0.0300 - acc: 0.9904 - val_loss: 0.0548 - val_acc: 0.9847
Epoch 20/20
5s - loss: 0.0281 - acc: 0.9905 - val_loss: 0.0586 - val_acc: 0.9849
Training duration : 120.80942297
10000/10000 [==============================] - 1s     
Network's test score [loss, accuracy]: [0.058565713905680807, 0.9849]

'''
