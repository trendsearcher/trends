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
from numpy.random import seed
seed(1)

#########
#from sklearn  import preprocessing
#poly = preprocessing.PolynomialFeatures(2)






df = pd.read_csv('C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded.csv', header= 0, error_bad_lines=False)
Y_set = df['average_grade']
X_set = df.drop(columns=['coeff_of_stand_deviat_after_norm', 'average_grade'])
index_list_of_whole_dataset = list(range(len(Y_set)))

group_of_items = index_list_of_whole_dataset               # a sequence or set will work here.
num_to_select = int(len(Y_set)/3)                           # set the number to select here.
list_of_random_items = random.sample(group_of_items, num_to_select)
list_of_test_index = list_of_random_items


Y_set = Y_set.values.tolist()
X_set = X_set.values.tolist()


original_grades = []
forecasted_grades = []
for i in range(num_to_select, (len(Y_set) - 500)):
    X_train = X_set[:i]
    Y_train = Y_set[:i]
    X_test = X_set[i:i + 2]
    Y_test = Y_set[i:i + 2]
    
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    Y_train = np.array([np.array(x) for x in Y_train])
    Y_test = np.array([int(x) for x in Y_test])

    
    
#    if Y_test[0] == 0:
#        original_grades.append([1,0])
#    else:
#        original_grades.append([0,1])
    model = tf.keras.models.Sequential()  # a basic feed-forward model
    model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.tanh))  # tanh
#    model.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))  #sigmoid
#    model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))  # relu
    model.add(tf.keras.layers.Dense(5, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # Good default optimizer to start with
    history  = model.fit(X_train, Y_train, epochs=2, validation_data=(X_test, Y_test))  # , callbacks=[plot_losses]
    forecasted_grades.append(model.predict([X_test])[0][0])
    
##
#
#
#
#val_loss, val_acc = model.evaluate(X_test, Y_test)  # evaluate the out of sample data with model
#print('************************************')
#print(val_loss)  # model's loss (error)
#print(val_acc)  # model's accuracy
#
#
#
#
##from tf.keras.models import load_model
##model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
##model = tf.keras.models.load_model('my_model.h5')
##
##
#
##new_model = tf.keras.models.load_model('epic_num_reader.model')
#prediction = model.predict([X_test])# дает распределение вероятностей
#print(prediction[:10])
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
