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
import csv
from tensorflow.keras.layers import  BatchNormalization# Dropout, Flatten, Activation,
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import roc_auc_score
picture_path1 = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_TPV_vectors_graded_20_waste.pkl'
numbers_path1 = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded_20_waste.csv'
waste_rating_path1 = 'C:\\Users\\user_PC\\Desktop\\sber\\samples_rating.csv'

picture_path2 =      'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_TPV_vectors_graded_all_20_waste.pkl'
numbers_path2 =      'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_graded_all_20_waste.csv'
waste_rating_path2 = 'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating.csv'

epochss = 100
auc_list = [0]*epochss
epoch_list = list(range(epochss))
acc_val = [0]*epochss
loss_list = [0]*epochss
val_loss_list = [0]*epochss
################################################################################
def get_list1():
    with open(waste_rating_path1, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        L = []
        for row in spamreader:
            L.append(ast.literal_eval(row[0]))
        L_onehot = []    
        for i in L:
            if i < np.mean(L):
                L_onehot.append(0)
            else:
                L_onehot.append(1)
        return(L_onehot)
def get_list2():
    with open(waste_rating_path2, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",")
        L = []
        for row in spamreader:
            L.append(ast.literal_eval(row[0]))
        L_onehot = []    
        for i in L:
            if i <= np.mean(L):
                L_onehot.append(0)
            else:
                L_onehot.append(1)
        return(L_onehot)
        
################################################################################
class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    def on_train_begin(self, logs={}):
        return
    def on_train_end(self, logs={}):
        return
    def on_epoch_begin(self, epoch, logs={}):
        return
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        y_pred = [x[1] for x in y_pred]
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val = [x[1] for x in y_pred_val]
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        auc_list[epoch] += roc_val
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return        
        
        
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

'''открываем датафреймы с картинками и числами'''
pic_df = pd.read_pickle(picture_path1)
Y_set_pic = pic_df['wasteness']
X_set_pic = pic_df.drop(columns=[ 'average_grade', 'wasteness'])
df = pd.read_csv(numbers_path1)
Y_set = df['wasteness'].values.tolist()
X_set = df.drop(columns=[ 'average_grade', 'wasteness'])
print(np.mean(Y_set))
aaa = (df[df['wasteness'] == 1]['average_grade'].values.tolist())
print(np.mean(aaa))


pic_df2 = pd.read_pickle(picture_path2)
Y_set_pic2 = pic_df2['wasteness']
X_set_pic2 = pic_df2.drop(columns=[ 'average_grade', 'wasteness'])
df2 = pd.read_csv(numbers_path2)
Y_set2 = df2['wasteness'].values.tolist()
X_set2 = df2.drop(columns=[ 'average_grade', 'wasteness', 'Unnamed: 0'])

print(np.mean(Y_set2))
bbb = (df2[df2['wasteness'] == 1]['average_grade'].values.tolist())
print(np.mean(bbb))
###########################################################################
###############################################################################
'''подготовка чисел на вход'''
X_train = X_set2.values.tolist()
Y_train = np.array([np.array(x) for x in Y_set2])
X_test = X_set.values.tolist()
Y_test = np.array([np.array(x) for x in Y_set])

X_train = np.array([np.array(x) for x in X_train])
X_test = np.array([np.array(x) for x in X_test])






##############################################################################
'''подготовка картинок на вход'''
X_train_pic = X_set_pic2.values.tolist()
X_train_picformated = []
for i in X_train_pic:
    vector = []
    for _ in i:
        subvector = ast.literal_eval(_)
        vector.append(subvector)
    X_train_picformated.append(vector)  
X_test_pic = X_set_pic.values.tolist()
X_test_picformated = []
for i in X_test_pic:
    vector = []
    for _ in i:
        subvector = ast.literal_eval(_)
        vector.append(subvector)
    X_test_picformated.append(vector)  
X_train_pic2 = X_set_pic.values.tolist()
X_train_picformated2 = []



print(np.mean(Y_train))
print(np.mean(Y_test))
###############################################################################
X_train_picformated = np.array([np.array(x) for x in X_train_picformated])
Y_train = np.array([np.array(x) for x in Y_train])
X_train_picformated = X_train_picformated.reshape(len(X_train_picformated),1000,7,1)#1000

X_test_picformated = np.array([np.array(x) for x in X_test_picformated])
Y_test = np.array([np.array(x) for x in Y_test])
X_test_picformated = X_test_picformated.reshape(len(X_test_picformated),1000,7,1)
################################################################################

correlation_list2 = []
AVERAGE_ACC_list2 = [] 
matrix2 = []

class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
model2 = tf.keras.models.Sequential()
model2.add(tf.keras.layers.Conv2D(50, kernel_size=(3, 7), activation='relu', input_shape=(1000,7,1)))#50
model2.add(tf.keras.layers.MaxPooling2D(pool_size=(3,1), strides=None, padding='valid', data_format='channels_last'))
model2.add(tf.keras.layers.Conv2D(25, kernel_size=(30,1), activation='relu'))#25  30
model2.add(tf.keras.layers.Conv2D(5, kernel_size=(60,1), activation='relu'))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(2, activation='softmax'))
model2.compile(optimizer= tf.train.AdamOptimizer(learning_rate=0.00015, beta1=0.9, beta2=0.99, epsilon=1e-08), loss='sparse_categorical_crossentropy', metrics=['accuracy'])#0.0.00005
history  = model2.fit(X_train_picformated, Y_train, epochs=epochss, validation_data=(X_test_picformated, Y_test),class_weight = class_weights,  callbacks=[roc_callback(training_data=(X_train_picformated, Y_train),validation_data=(X_test_picformated, Y_test))] )# 
for indx, (i, j, k) in enumerate(zip(history.history['val_acc'], history.history['loss'], history.history['val_loss'])):
    loss_list[indx]  += j
    val_loss_list[indx] +=k

auc_list = [x for x in auc_list]

lines = plt.plot(epoch_list, auc_list)
l2= lines
plt.setp(lines, linestyle='-')
plt.setp(l2, linewidth=1, color='r')
plt.title('auc-red' )
plt.grid()
plt.show()
plt.pause(0.05)


#print(roc_auc_score(Y_test, matrix))

 
