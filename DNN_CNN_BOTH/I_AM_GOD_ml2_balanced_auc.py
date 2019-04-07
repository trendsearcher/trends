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
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
################
iterations = 1
epochss = 2
###############
auc_list = [0]*epochss

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





epoch_list = list(range(epochss))
acc_val = [0]*epochss
loss_list = [0]*epochss
val_loss_list = [0]*epochss
for i in range (iterations):

    df = pd.read_csv('C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded.csv', header= 0, error_bad_lines=False)
    Y_set = df['average_grade']
    X_set = df.drop(columns=[ 'average_grade'])
    index_list_of_whole_dataset = list(range(len(Y_set)))
    
    group_of_items = index_list_of_whole_dataset               # a sequence or set will work here.
    num_to_select = int(len(Y_set)/5)                           # set the number to select here.
    list_of_random_items = random.sample(group_of_items, num_to_select)
    list_of_test_index = list_of_random_items
    list_of_train_index = [x for x in index_list_of_whole_dataset if x not in list_of_test_index]
    
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
    
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    
    
    model = tf.keras.models.Sequential()  # a basic feed-forward model   
    model.add(tf.keras.layers.Dense(500, activation=tf.nn.relu))  # tanh
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.sigmoid))  # our output layer. 10 units for 10 classes. Softmax for probability distribution
    model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # Good default optimizer to start with
    history  = model.fit(X_train, Y_train, epochs=epochss, validation_data=(X_test, Y_test), class_weight=class_weights, callbacks=[roc_callback(training_data=(X_train, Y_train),validation_data=(X_test, Y_test))])#, 
    print(model.summary())
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




