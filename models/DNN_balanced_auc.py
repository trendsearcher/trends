# -*- coding: utf-8 -*-
"""
script reads dataframe that contains numerical features
 and grades up/down(+waste)
can we get more than 50% acc in average?
actually we can)
"""
import tensorflow as tf  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.optimizers import  SGD
from sklearn.model_selection import train_test_split

################
iterations = 20
epochss = 70
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

df = pd.read_csv('C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_graded_all2.csv', header= 0, error_bad_lines=False)

Y_set = df['average_grade']
X_set = df.drop(columns=[ 'average_grade', 'wasteness'])
X_set = X_set.values.tolist()
Y_set = Y_set.values.tolist()
X_set = np.array([np.array(x) for x in X_set])
Y_set = np.array([np.array(x) for x in Y_set])

epoch_list = list(range(epochss))
acc_val = [0]*epochss
loss_list = [0]*epochss
val_loss_list = [0]*epochss

for i in range (iterations):
    X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=i)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    
    model = tf.keras.models.Sequential()  
    model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))  
    model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.000025, beta1=0.9, beta2=0.99, epsilon=1e-08), loss='sparse_categorical_crossentropy',metrics=['accuracy'])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # SGD(momentum=0.99, nesterov=True)
    history  = model.fit(X_train, Y_train, epochs=epochss, validation_data=(X_test, Y_test),class_weight=class_weights, callbacks=[roc_callback(training_data=(X_train, Y_train),validation_data=(X_test, Y_test))])#, class_weight=class_weights
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




