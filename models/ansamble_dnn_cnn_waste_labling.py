# -*- coding: utf-8 -*-
"""
just an experiment of labling good/bad examples by the frequency of guessing by 
ansamble of models. writes result in list (csv)
nothing to see here yet
@author: user_PC
"""
import csv
import tensorflow as tf 
import numpy as np
import pandas as pd
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
from sklearn.model_selection import train_test_split

picture_path = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_TPV_vectors_graded.pkl'
numbers_path = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded.csv'
waste_grades_path = 'C:\\Users\\user_PC\\Desktop\\sber\\samples_rating12.csv'
#################################################################################################
iterations = 200
###############
###############
gades1 = []
gades2 = []
def similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i == j for i, j in zip(a, b))
def similarity2(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i == j for i, j in zip(a, b) if i ==1 and j ==1)
class roc_callback(Callback):
    def __init__(self,training_data,validation_data, pred_list):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.averageweights = pred_list
    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val2 = [round(x[1]) for x in y_pred_val]
#        if similarity(y_pred_val2, self.y_val)/len(self.y_val) > 0.5:
        onehot_y_pred_correct = []
        for i,j in zip(y_pred_val2, self.y_val):
            if i == j:
                onehot_y_pred_correct.append(1)
            else:
                onehot_y_pred_correct.append(0)
        self.averageweights.append(onehot_y_pred_correct)
        return
    
pic_df = pd.read_pickle(picture_path)

Y_set = pic_df['average_grade']
X_set = pic_df.drop(columns=[ 'average_grade'])
X_set = X_set.values.tolist()
X_set_formated = []
for i in X_set:
    vector = []
    for _ in i:
        subvector = ast.literal_eval(_)
        vector.append(subvector)
    X_set_formated.append(vector)  
Y_set = Y_set.values.tolist()
X_set_formated = np.array([np.array(x) for x in X_set_formated])
Y_set = np.array([np.array(x) for x in Y_set])

df = pd.read_csv(numbers_path)
Y_set = df['average_grade']
X_set = df.drop(columns=[ 'average_grade'])
X_set = X_set.values.tolist()
Y_set = Y_set.values.tolist()
X_set = np.array([np.array(x) for x in X_set])
Y_set = np.array([np.array(x) for x in Y_set])


grabage_and_gold_list = np.array([0]*len(X_set))
for i in range (iterations):
    
    X_train_picformated, X_test_picformated = train_test_split(X_set_formated,  test_size=0.2, random_state=i+25)
    list_of_train_index, list_of_test_index = train_test_split(list(range(len(X_set))),  test_size=0.2, random_state=i+25)
    X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=i+25)
    ###########################################################################
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    ###############################################################################
    X_train_picformated = X_train_picformated.reshape(len(X_train_picformated),1000,7,1)#1000
    X_test_picformated = X_test_picformated.reshape(len(X_test_picformated),1000,7,1)

    list_of_predicted_pos_by_model_general = [0]*(len(Y_train)+len(Y_test))
    matrix1 = []
    matrix11 = np.array([0]*len(Y_test))
    model1 = tf.keras.models.Sequential()     
    model1.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))  
    model1.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  

    model1.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.000075, beta1=0.9, beta2=0.99, epsilon=1e-08), loss='sparse_categorical_crossentropy',metrics=['accuracy'])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # SGD(momentum=0.99, nesterov=True)
    history1  = model1.fit(X_train, Y_train, epochs=70, validation_data=(X_test, Y_test),class_weight=class_weights, callbacks=[roc_callback(training_data=(X_train, Y_train),validation_data=(X_test, Y_test), pred_list = matrix1)])#, class_weight=class_weights
    for i in matrix1:
        matrix11 += np.array(i)
    matrix11 = [x/len(matrix1) for x in matrix11] 
    
    matrix2 = []
    matrix22 = np.array([0]*len(Y_test))
    model2 = tf.keras.models.Sequential()
    model2.add(tf.keras.layers.Conv2D(50, kernel_size=(3, 7), activation='relu', input_shape=(1000,7,1)))#50
    model2.add(tf.keras.layers.MaxPooling2D(pool_size=(3,1), strides=None, padding='valid', data_format='channels_last'))
    model2.add(tf.keras.layers.Conv2D(25, kernel_size=(30,1), activation='relu'))#25  30
    model2.add(tf.keras.layers.Conv2D(5, kernel_size=(60,1), activation='relu'))
    model2.add(tf.keras.layers.Flatten())
    model2.add(tf.keras.layers.Dense(2, activation='softmax'))
    model2.compile(optimizer= tf.train.AdamOptimizer(learning_rate=0.00003, beta1=0.9, beta2=0.99, epsilon=1e-08), loss='sparse_categorical_crossentropy', metrics=['accuracy'])#0.00003
    history2  = model2.fit(X_train_picformated, Y_train, epochs=30, validation_data=(X_test_picformated, Y_test),  class_weight=class_weights, callbacks=[roc_callback(training_data=(X_train_picformated, Y_train),validation_data=(X_test_picformated, Y_test), pred_list = matrix2)])# 
    for i in matrix2:
        matrix22 += np.array(i)
    matrix22 = [x/len(matrix2) for x in matrix22] 
    
    matrix = []#[round(x)for x in matrix1]##############################
    for i, j in zip(matrix11, matrix22):
        average_possibility = round((i+j)/2)
        matrix.append(average_possibility)
    for i, j in zip(list_of_test_index, matrix):
        list_of_predicted_pos_by_model_general[i] = j
    list_of_predicted_pos_by_model_general = np.array(list_of_predicted_pos_by_model_general)    
    
    
    grabage_and_gold_list = np.add(grabage_and_gold_list, list_of_predicted_pos_by_model_general)
print(sum(grabage_and_gold_list))
np.savetxt(waste_grades_path, grabage_and_gold_list, delimiter=",")

