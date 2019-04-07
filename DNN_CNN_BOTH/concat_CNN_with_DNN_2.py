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
picture_path = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_TPV_vectors_graded.pkl'
numbers_path = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded.csv'
#################################################################################################
iterations = 100
epochss = 20
###############
auc_list = [0]*epochss
epoch_list = list(range(epochss))
acc_val = [0]*epochss
loss_list = [0]*epochss
val_loss_list = [0]*epochss
###############
auc_list = [0]*epochss
epoch_list = list(range(epochss))
acc_val = [0]*epochss
loss_list = [0]*epochss
val_loss_list = [0]*epochss
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
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val = [x[1] for x in y_pred_val]
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        auc_list[epoch] += roc_val
        print('\roc-auc_val: %s' % (str(round(roc_val,4))),end=100*' '+'\n')
        return
    def on_batch_begin(self, batch, logs={}):
        return
    def on_batch_end(self, batch, logs={}):
        return




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
    X_train = X_train.reshape(len(X_train),35,1)
    X_test = X_test.reshape(len(X_test),35,1)     
    
    
    
    visible1 = tf.keras.layers.Input(shape=(35,1))
    visible2 = tf.keras.layers.Input(shape=(1000,7,1))
    # first feature extractor
    deep_n = tf.keras.layers.Dense(20, activation='relu')(visible1)
    normas1 = tf.keras.layers.BatchNormalization()(deep_n)
    flat1 = tf.keras.layers.Flatten()(normas1)
    # second feature extractor
    conv2 = tf.keras.layers.Conv2D(50, kernel_size=(100, 7), activation='relu')(visible2)
    normas2 = tf.keras.layers.BatchNormalization()(conv2)
    conv3 = tf.keras.layers.Conv2D(25, kernel_size=(30, 1), activation='relu')(normas2)
    normas3 = tf.keras.layers.BatchNormalization()(conv3)
    flat2 = tf.keras.layers.Flatten()(normas3)
    
    # merge feature extractors
    merge = tf.keras.layers.concatenate([flat1, flat2])
    # interpretation layer
    hidden1 = tf.keras.layers.Dense(10, activation='relu')(merge)
    # prediction output
    output = tf.keras.layers.Dense(2, activation='sigmoid')(hidden1)
    model = tf.keras.models.Model(inputs=[visible1,visible2], outputs=output)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.train.AdamOptimizer(learning_rate=0.000008, beta1=0.9, beta2=0.99, epsilon=1e-08), metrics = ['accuracy'])
    history = model.fit([X_train, X_train_picformated], Y_train, class_weight=class_weights, nb_epoch = epochss, verbose = 1,validation_data=([X_test, X_test_picformated], Y_test), callbacks=[roc_callback(training_data=([X_test, X_test_picformated], Y_train),validation_data=([X_test, X_test_picformated], Y_test))])#
    
    for indx, (i, j, k) in enumerate(zip(history.history['val_acc'], history.history['loss'], history.history['val_loss'])):
        acc_val[indx] += i 
        loss_list[indx]  += j
        val_loss_list[indx] +=k
        
acc_val = [x/iterations for x in acc_val]
auc_list = [x/iterations for x in auc_list]
loss_list = [x/iterations for x in loss_list]
val_loss_list = [x/iterations for x in val_loss_list]

lines = plt.plot(epoch_list, acc_val, epoch_list, auc_list)
l1, l2= lines
plt.setp(lines, linestyle='-')
plt.setp(l1, linewidth=1, color='b')
plt.setp(l2, linewidth=1, color='r')
plt.title('acc-blue, auc-red' )
plt.grid()
plt.show()
plt.pause(0.05)

lines = plt.plot(epoch_list, loss_list,epoch_list, val_loss_list)
l1, l2= lines
plt.setp(lines, linestyle='-')
plt.setp(l1, linewidth=1, color='g')
plt.setp(l2, linewidth=1, color='y')
plt.title('train_loss-green, test_loss -yellow' )
plt.grid()
plt.show()





 
