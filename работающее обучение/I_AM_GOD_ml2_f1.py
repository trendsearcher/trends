# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:26:53 2019

@author: user_PC
"""
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
#from numpy.random import seed
#seed(1)
iterations = 50
epochss = 60
#########

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


class PlotLosses(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.grid()
        plt.show();
plot_losses = PlotLosses()


epoch_list = list(range(epochss))
acc_val = [0]*epochss
f1_list = [0]*epochss
for i in range (iterations):

    df = pd.read_csv('C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded.csv', header= 0, error_bad_lines=False)
    Y_set = df['average_grade']
    X_set = df.drop(columns=[ 'average_grade'])
    index_list_of_whole_dataset = list(range(len(Y_set)))
    
    group_of_items = index_list_of_whole_dataset               # a sequence or set will work here.
    num_to_select = int(len(Y_set)/6)                           # set the number to select here.
    list_of_random_items = random.sample(group_of_items, num_to_select)
    list_of_test_index = list_of_random_items
    list_of_train_index = [x for x in index_list_of_whole_dataset if x not in  list_of_test_index]
    
    X_train = X_set.loc[list_of_train_index].values.tolist()
    X_test = X_set.loc[list_of_test_index].values.tolist()
    #print( Y_set.loc[list_of_train_index].describe())
    Y_train = Y_set.loc[list_of_train_index].values.tolist()
    #print( Y_set.loc[list_of_test_index].describe())
    Y_test= Y_set.loc[list_of_test_index].values.tolist()
    #time.sleep(10)
    
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    Y_train = np.array([np.array(x) for x in Y_train])
    Y_test = np.array([np.array(x) for x in Y_test])
    
    
    model = tf.keras.models.Sequential()  # a basic feed-forward model
    model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
    model.add(tf.keras.layers.Dense(1000, activation=tf.nn.relu))  # tanh
#    model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  #sigmoid
#    model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))  # relu
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy', f1])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # Good default optimizer to start with
    history  = model.fit(X_train, Y_train, epochs=epochss, validation_data=(X_test, Y_test))#, , callbacks=[plot_losses]
    print(history.history)
    for indx, (i, j) in enumerate(zip(history.history['val_acc'],history.history['val_f1'])):
        acc_val[indx] += i 
        f1_list[indx] += j
acc_val = [x/iterations for x in acc_val]
f1_list = [x/iterations for x in f1_list]


lines = plt.plot(epoch_list, acc_val, epoch_list, f1_list)
l1, l2= lines
plt.setp(lines, linestyle='-')
plt.setp(l1, linewidth=1, color='b')
plt.setp(l2, linewidth=1, color='r')
plt.title('acc-blue, f1-red')
plt.grid()
plt.show()





#plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.grid()
#plt.show()
#
#



#val_loss, val_acc = model.evaluate(X_test, Y_test)  # evaluate the out of sample data with model
#print('************************************')
#print(val_loss)  # model's loss (error)
#print(val_acc)  # model's accuracy




#from tf.keras.models import load_model

#model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#model = tf.keras.models.load_model('my_model.h5')
#
#

#new_model = tf.keras.models.load_model('epic_num_reader.model')
#prediction = new_model.predict([X_test])# дает распределение вероятностей
#print(np.argmax(prediction(0)))

















#

#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
##mnist = tf.keras.datasets.mnist
###download mnist data and split into train and test sets
##(X_train, y_train), (X_test, y_test) = mnist.load_data()
##import matplotlib.pyplot as plt
###plot the first image in the dataset
##plt.imshow(X_train[0])
###check image shape
##X_train[0].shape
##X_train = X_train.reshape(60000,28,28,1)
##print(X_train[0])
##X_test = X_test.reshape(10000,28,28,1)
###one-hot encode target column
##y_train = tf.keras.utils.to_categorical(y_train)
##y_test = tf.keras.utils.to_categorical(y_test)
##y_train[0]
###from keras.models import Sequential
###from keras.layers import Dense, Conv2D, Flatten
###create model
##model = tf.keras.models.Sequential()
###add model layers
##model.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
##model.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
##model.add(tf.keras.layers.Flatten())
##model.add(tf.keras.layers.Dense(10, activation='softmax'))
###compile model using accuracy to measure model performance
##model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
###train the model
##model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
###predict first 4 images in the test set
##model.predict(X_test[:4])
###actual results for first 4 images in test set
##y_test[:4]
##
##
