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

#########

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


epoch_list = list(range(100))
acc_val = [0]*100
for i in range (20):

    df1 = pd.read_csv('C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded.csv', header= 0, error_bad_lines=False)
    df2 = pd.read_csv('C:\\Users\\user_PC\\Desktop\\rts\\normal_trends_outofdublers_norm_graded.csv', header= 0, error_bad_lines=False)
    Y_set1 = df1['average_grade']
    X_set1 = df1.drop(columns=[ 'average_grade'])
    
    Y_set2 = df2['average_grade']
    X_set2 = df2.drop(columns=[ 'average_grade'])
    
    X_train = X_set1.values.tolist()
    X_test = X_set2.values.tolist()
    #print( Y_set.loc[list_of_train_index].describe())
    Y_train = Y_set1.values.tolist()
    #print( Y_set.loc[list_of_test_index].describe())
    Y_test= Y_set2.values.tolist()
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
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # Good default optimizer to start with
    history  = model.fit(X_train, Y_train, epochs=100, validation_data=(X_test, Y_test))#, , callbacks=[plot_losses]
    for indx, i in enumerate(history.history['val_acc']):
        acc_val[indx] += i 

acc_val = [x/20 for x in acc_val]
plt.plot(epoch_list, acc_val)
plt.title('model accuracy')
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
