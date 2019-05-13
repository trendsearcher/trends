# -*- coding: utf-8 -*-
"""
just experiment. is there any synergy in 2 types of data combined in merged dnn/cnn in average?
dispite othogonality...actually no
"""
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
picture_path = 'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_TPV_vectors_graded_all2.pkl'
numbers_path = 'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_graded_all2.csv'
#################################################################################################
################
iterations = 50
epochss = 40
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
        y_pred = self.model.predict(self.x)
        y_pred = [x[1] for x in y_pred]
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        y_pred_val = [x[1] for x in y_pred_val]
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        auc_list[epoch] += roc_val
        print('\roc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
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
    X_train, X_test, Y_train, Y_test  = train_test_split(X_set, Y_set,test_size=0.2, random_state=i)
    X_train_pic, X_test_pic, Y_train_pic, Y_test_pic  = train_test_split(X_set_pic, Y_set_pic, test_size=0.2, random_state=i)
    ###############################################################################
    '''подготовка чисел на вход'''
    X_train = X_train.values.tolist()
    X_test = X_test.values.tolist()
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    ##############################################################################
    '''подготовка картинок на вход'''
    X_train_pic = X_train_pic.values.tolist()
    X_train_picformated = []
    for i in X_train_pic:
        vector = []
        for _ in i:
            subvector = ast.literal_eval(_)
            vector.append(subvector)
        X_train_picformated.append(vector)  
    X_test_pic = X_test_pic.values.tolist()
    X_test_picformated = []
    for i in X_test_pic:
        vector = []
        for _ in i:
            subvector = ast.literal_eval(_)
            vector.append(subvector)
        X_test_picformated.append(vector)  
    Y_train = Y_train.values.tolist()
    Y_test= Y_test.values.tolist()
    ###############################################################################
    X_train_picformated = np.array([np.array(x) for x in X_train_picformated])
    X_test_picformated = np.array([np.array(x) for x in X_test_picformated])
    Y_train = np.array([np.array(x) for x in Y_train])
    Y_test = np.array([np.array(x) for x in Y_test])
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    ###############################################################################
    X_train_picformated = X_train_picformated.reshape(len(X_train_picformated),1000,7,1)#1000
    X_test_picformated = X_test_picformated.reshape(len(X_test_picformated),1000,7,1)
    X_train = X_train.reshape(len(X_train),len(X_train[0]),1)
    X_test = X_test.reshape(len(X_test),len(X_test[0]),1)
     
    
    
    
    visible1 = tf.keras.layers.Input(shape=(len(X_train[0]),1))
    visible2 = tf.keras.layers.Input(shape=(1000,7,1))
    # first feature extractor
#    deep_n = tf.keras.layers.Dense(20, activation='relu')(visible1)
    flat1 = tf.keras.layers.Flatten()(visible1)
    # second feature extractor
    conv1 = tf.keras.layers.Conv2D(50, kernel_size=(3, 7), activation='relu')(visible2)
    conv2 = tf.keras.layers.Conv2D(25, kernel_size=(30, 1), activation='relu')(conv1)
    conv3 = tf.keras.layers.Conv2D(5, kernel_size=(60, 1), activation='relu')(conv2)
    flat2 = tf.keras.layers.Flatten()(conv3)
    
    
    # merge feature extractors
    merge = tf.keras.layers.concatenate([flat1, flat2])
    # interpretation layer
    hidden1 = tf.keras.layers.Dense(50, activation='relu')(merge)
    # prediction output
    output = tf.keras.layers.Dense(2, activation='sigmoid')(hidden1)
    model = tf.keras.models.Model(inputs=[visible1,visible2], outputs=output)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.train.AdamOptimizer(learning_rate=0.00001, beta1=0.9, beta2=0.99, epsilon=1e-08), metrics = ['accuracy'])
    history = model.fit([X_train, X_train_picformated], Y_train, class_weight=class_weights, nb_epoch = epochss, validation_data=([X_test, X_test_picformated], Y_test))
    
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





 
