# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:10:16 2019

@author: user_PC
"""
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np
import pandas as pd
import random
import tensorflow.keras.backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
import ast
from tensorflow.keras.layers import  BatchNormalization# Dropout, Flatten, Activation,
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
#from tensorflow.keras.layers import  merge
picture_path = 'C:\\Users\\user_PC\\Desktop\\sber2\\normal_trends_outofdublers_norm_TPV_vectors_graded_minmax.pkl'
numbers_path = 'C:\\Users\\user_PC\\Desktop\\sber2\\normal_trends_outofdublers_norm_graded_minmax.csv'
#################################################################################################
iterations = 60
###############
###############
gades1 = []
gades2 = []
sumgrades = []
def similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i == j for i, j in zip(a, b))
def similarity2(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i == j for i, j in zip(a, b) if i ==1 and j ==1)
class roc_callback1(Callback):
    def __init__(self,training_data,validation_data, pred_list):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.corr = pred_list
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val = np.array([x[1] for x in y_pred_val])
        if epoch >= 55 and epoch <=56: 
            self.corr.append(y_pred_val)
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return
class roc_callback2(Callback):
    def __init__(self,training_data,validation_data, pred_list):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.corr = pred_list
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val = np.array([x[1] for x in y_pred_val])
        if epoch >= 94 and epoch <=95: 
            self.corr.append(y_pred_val)
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return

THE_RESULT = []
for i in range (iterations):
    '''открываем датафреймы с картинками и числами'''
    pic_df = pd.read_pickle(picture_path)
    Y_set_pic = pic_df['average_grade']
    X_set_pic = pic_df.drop(columns=[ 'average_grade'])
    df = pd.read_csv(numbers_path)
    Y_set = df['average_grade']
    X_set = df.drop(columns=[ 'average_grade'])
    ###########################################################################
    '''генерация случайной выборки'''
    index_list_of_whole_dataset = list(range(len(Y_set_pic)))
    group_of_items = index_list_of_whole_dataset               # a sequence or set will work here.
    num_to_select = int(len(Y_set_pic)/5)                           # set the number to select here.
    list_of_random_items = random.sample(group_of_items, num_to_select)
    list_of_test_index = list_of_random_items
    list_of_train_index = [x for x in index_list_of_whole_dataset if x not in list_of_test_index]
    ###############################################################################
    '''подготовка чисел на вход'''
    X_train = X_set.loc[list_of_train_index].values.tolist()
    X_test = X_set.loc[list_of_test_index].values.tolist()
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    ##############################################################################
    '''подготовка картинок на вход'''
    X_train_pic = X_set_pic.loc[list_of_train_index].values.tolist()
    X_train_picformated = []
    for i in X_train_pic:
        vector = []
        for _ in i:
            subvector = ast.literal_eval(_)
            vector.append(subvector)
        X_train_picformated.append(vector)  
    X_test_pic = X_set_pic.loc[list_of_test_index].values.tolist()
    X_test_picformated = []
    for i in X_test_pic:
        vector = []
        for _ in i:
            subvector = ast.literal_eval(_)
            vector.append(subvector)
        X_test_picformated.append(vector)  
    Y_train = Y_set_pic.loc[list_of_train_index].values.tolist()
    Y_test= Y_set_pic.loc[list_of_test_index].values.tolist()
    ###############################################################################
    X_train_picformated = np.array([np.array(x) for x in X_train_picformated])
    X_test_picformated = np.array([np.array(x) for x in X_test_picformated])
    Y_train = np.array([np.array(x) for x in Y_train])
    Y_test = np.array([np.array(x) for x in Y_test])
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    ###############################################################################
    X_train_picformated = X_train_picformated.reshape(len(X_train_picformated),1000,7,1)#1000
    X_test_picformated = X_test_picformated.reshape(len(X_test_picformated),1000,7,1)
#    X_train = X_train.reshape(len(X_train),28,1)
#    X_test = X_test.reshape(len(X_test),28,1)     
    correlation_list1 = []
    AVERAGE_ACC_list1 = []
    matrix1 = []
    for _ in range(1):
        model1 = tf.keras.models.Sequential()  # a basic feed-forward model   
        model1.add(tf.keras.layers.Dense(50, activation=tf.nn.relu))  # tanh
        model1.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution
        model1.compile(optimizer=SGD(momentum=0.99, nesterov=True), loss='sparse_categorical_crossentropy',metrics=['accuracy'])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # SGD(momentum=0.99, nesterov=True)
        history1  = model1.fit(X_train, Y_train, epochs=56, validation_data=(X_test, Y_test),class_weight=class_weights, callbacks=[roc_callback1(training_data=(X_train, Y_train),validation_data=(X_test, Y_test), pred_list = matrix1)])#, class_weight=class_weights
        AVERAGE_ACC_list1.append(history1.history['val_acc'][55])
    matrix11 = [0]*len(Y_test)
    for i in matrix1:
        matrix11 += np.array(i)
    matrix1 = [x/len(matrix1) for x in matrix11]
        
    correlation_list2 = []
    AVERAGE_ACC_list2 = [] 
    matrix2 = []
    for _ in range(1):
        model2 = tf.keras.models.Sequential()
        model2.add(tf.keras.layers.Conv2D(50, kernel_size=(3, 7), activation='relu', input_shape=(1000,7,1)))#50
        model2.add(tf.keras.layers.Conv2D(25, kernel_size=(30,1), activation='relu'))#25  30
        model2.add(tf.keras.layers.Conv2D(5, kernel_size=(60,1), activation='relu'))
        model2.add(tf.keras.layers.Flatten())
        model2.add(tf.keras.layers.Dense(2, activation='softmax'))
        model2.compile(optimizer= tf.train.AdamOptimizer(learning_rate=0.00003, beta1=0.9, beta2=0.99, epsilon=1e-08), loss='sparse_categorical_crossentropy', metrics=['accuracy'])#0.00003
        history2  = model2.fit(X_train_picformated, Y_train, epochs=95, validation_data=(X_test_picformated, Y_test),  class_weight=class_weights, callbacks=[roc_callback2(training_data=(X_train_picformated, Y_train),validation_data=(X_test_picformated, Y_test), pred_list = matrix2)] )# 
        AVERAGE_ACC_list2.append(history2.history['val_acc'][94])
    matrix22 = [0]*len(Y_test)
    for i in matrix2:
        matrix22 += np.array(i)
    matrix2 = [x/len(matrix2) for x in matrix22] 

    AVERAGE_ACC1 = np.mean(AVERAGE_ACC_list1)
    AVERAGE_ACC2 = np.mean(AVERAGE_ACC_list2)
    gades1.append(AVERAGE_ACC1)
    gades2.append(AVERAGE_ACC2)
    matrix = []
    for i,j in zip(matrix1, matrix2):
        matrix.append(round((i+j)/2))
    sumgrades.append(similarity(matrix, Y_test)/len(Y_test))
    
print(gades1)
print(np.mean(gades1))
print()
print(gades2)
print(np.mean(gades2))
print()
print(sumgrades)
print(np.mean(sumgrades))

 
