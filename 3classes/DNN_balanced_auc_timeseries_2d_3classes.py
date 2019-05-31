# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:26:53 2019

@author: user_PC
"""
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow.keras.backend as K
from itertools import cycle
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, roc_curve, auc
from tensorflow.keras.callbacks import Callback, EarlyStopping
import ast
from scipy import interp
#from tensorflow.keras.layers import  BatchNormalization#, Dense, Dropout, Flatten, Activation,
from sklearn.model_selection import train_test_split
################
iterations = 1
epochss = 90
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
    def on_epoch_begin(self, epoch, logs={}):
#        n_classes = Y_test.shape[1]
#        y_score = self.model.predict(X_test_formated)
#        y_test = self.y_val
#        # Compute ROC curve and ROC area for each class
#        fpr = dict()
#        tpr = dict()
#        roc_auc = dict()
#        for i in range(n_classes):
#            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#            roc_auc[i] = auc(fpr[i], tpr[i])
#
#        # Compute micro-average ROC curve and ROC area
#        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
#        # First aggregate all false positive rates
#        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#        lw = 2
#        # Then interpolate all ROC curves at this points
#        mean_tpr = np.zeros_like(all_fpr)
#        for i in range(n_classes):
#            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#        # Finally average it and compute AUC
#        mean_tpr /= n_classes
#        
#        fpr["macro"] = all_fpr
#        tpr["macro"] = mean_tpr
#        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
#        # Plot all ROC curves
#        plt.figure()
#        plt.plot(fpr["micro"], tpr["micro"],
#         label='micro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["micro"]),
#         color='deeppink', linestyle=':', linewidth=4)
#        
#        plt.plot(fpr["macro"], tpr["macro"],
#         label='macro-average ROC curve (area = {0:0.2f})'
#               ''.format(roc_auc["macro"]),
#         color='navy', linestyle=':', linewidth=4)
#        
#        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#        for i, color in zip(range(n_classes), colors):
#            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#                     label='ROC curve of class {0} (area = {1:0.2f})'
#                     ''.format(i, roc_auc[i]))
#        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#        plt.xlim([0.0, 1.0])
#        plt.ylim([0.0, 1.05])
#        plt.xlabel('False Positive Rate')
#        plt.ylabel('True Positive Rate')
#        plt.title('Some extension of Receiver operating characteristic to multi-class')
#        plt.legend(loc="lower right")
#        plt.show()
        
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        auc_list[epoch] += roc_val
        print('\roc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    

df = pd.read_csv('C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_graded_all_3class_10.csv')
#df = df[df['wasteness'] >= 0.9]

Y_set = df['average_grade']
#ave_val = np.mean(Y_set.values.tolist())
#print(ave_val)
X_set = df.drop(columns=[ 'average_grade', 'money', 'Unnamed: 0'])
X_set = X_set.values.tolist()
Y_set = Y_set.values.tolist()
X_set = np.array([np.array(x) for x in X_set])
Y_set = np.array([np.array(x) for x in Y_set])

epoch_list = list(range(epochss))
acc_val = [0]*epochss
loss_list = [0]*epochss
val_loss_list = [0]*epochss
for i in range (iterations):
    X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.02, random_state=i)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    print(class_weights)

    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=3, dtype='float32')
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=3, dtype='float32')

    
    model = tf.keras.models.Sequential()  # a basic feed-forward model   
    model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu))  # tanh
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.000025, beta1=0.9, beta2=0.99, epsilon=1e-08), loss='categorical_crossentropy',metrics=['accuracy'])#=tf.train.AdamOptimizer(learning_rate=0.000005, beta1=0.9, beta2=0.99, epsilon=1e-08),   # SGD(momentum=0.99, nesterov=True)
    history  = model.fit(X_train, Y_train, epochs=epochss, validation_data=(X_test, Y_test),class_weight=class_weights, callbacks=[roc_callback(training_data=(X_train, Y_train),validation_data=(X_test, Y_test))])#, class_weight=class_weights
    ###############################################################################
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

#n_classes = Y_test.shape[1]
#y_score = model.predict(X_test)
#y_test = Y_test
## Compute ROC curve and ROC area for each class
#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#for i in range(n_classes):
#    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
#    roc_auc[i] = auc(fpr[i], tpr[i])
## Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
## First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#lw = 2
## Then interpolate all ROC curves at this points
#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
#    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
## Finally average it and compute AUC
#mean_tpr /= n_classes
#
#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
## Plot all ROC curves
#plt.figure()
#plt.plot(fpr["micro"], tpr["micro"],
# label='micro-average ROC curve (area = {0:0.2f})'
#       ''.format(roc_auc["micro"]),
# color='deeppink', linestyle=':', linewidth=4)
#
#plt.plot(fpr["macro"], tpr["macro"],
# label='macro-average ROC curve (area = {0:0.2f})'
#       ''.format(roc_auc["macro"]),
# color='navy', linestyle=':', linewidth=4)
#
#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
#for i, color in zip(range(n_classes), colors):
#    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#             label='ROC curve of class {0} (area = {1:0.2f})'
#             ''.format(i, roc_auc[i]))
#
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Some extension of Receiver operating characteristic to multi-class')
#plt.legend(loc="lower right")
#plt.show()



#ВНЕШНЯЯ ПРОВЕРКА НА 2019 ГОД

output_numbers = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded_all_3class.csv'
df2 = pd.read_csv(output_numbers)
Y_set2 = df2['average_grade']
#ave_val = np.mean(Y_set.values.tolist())
#print(ave_val)
X_set2 = df2.drop(columns=['average_grade'])
X_set2 = X_set2.values.tolist()
Y_set2 = Y_set2.values.tolist()
X_set2 = np.array([np.array(x) for x in X_set2])
Y_set2 = np.array([np.array(x) for x in Y_set2])
#one-hot encode target column
Y_test2 = tf.keras.utils.to_categorical(Y_set2, num_classes=3, dtype='float32')



n_classes = 3
y_score = model.predict(X_set2)
y_test = Y_test2
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
lw = 2
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
# Plot all ROC curves
plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
 label='micro-average ROC curve (area = {0:0.2f})'
       ''.format(roc_auc["micro"]),
 color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
 label='macro-average ROC curve (area = {0:0.2f})'
       ''.format(roc_auc["macro"]),
 color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
#





