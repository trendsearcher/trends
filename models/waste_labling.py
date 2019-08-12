# -*- coding: utf-8 -*-
"""
скрипт для выставления оценки 'эталонности' примеров из датасета.
для выставления оценки делается множество рандомных сплитов датасета пополам
для каждого сплита делается попытка найти эталоны, им присваивается оценка 1, 
затем делается следующий сплит.
"""

# -*- coding: utf-8 -*-
"""
just an experiment of labling good/bad examples by the frequency of guessing by models
nothing to see here yet
@author: user_PC
"""
import csv
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
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

picture_path = 'C:\\Users\\user_PC\\Desktop\\sber3\\normal_trends_outofdublers_norm_TPV_vectors_graded_20.pkl'
waste_grades_path = 'C:\\Users\\user_PC\\Desktop\\sber3\\samples_rating_new_2.csv'
#picture_path = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_TPV_vectors_graded_20.pkl'
#numbers_path = 'C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded_20.csv'
#waste_grades_path = 'C:\\Users\\user_PC\\Desktop\\sber\\samples_rating05.csv'

#################################################################################################
iterations = 400
accuracy_treshold = 0.53
roc_treshold = 0.53
###############
def similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i == j for i, j in zip(a, b))
def similarity2(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i == j for i, j in zip(a, b) if i ==1 and j ==1)
class waste_callback(Callback):
    '''на эпохе, при которой достигается максимум средних значений аук на 
      кроссвалидации,если на валидации метрики аккураси и аук больше граничных,
      то смотрим на модел.предикт всего датасата и отбираем в качестве эталонов
      10% от тех примеров, в которых она больше всего уверена и тем из 10%,
      которые были классифицированы правильно присваиваем 1, остальным - 0'''
    def __init__(self,training_data,validation_data, alldata, pred_list):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.alldata_x = alldata[0]
        self.alldata_y = alldata[1]

        self.averageweights = pred_list

    def on_epoch_end(self, epoch, logs={}):
        y_pred_val = self.model.predict(self.x_val)
        y_hat_round_val = [round(x[1]) for x in y_pred_val]
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print(' - roc_val = {}'.format(roc_val))

        if (epoch == 15
           and similarity(y_hat_round_val, [x[1] for x in self.y_val])/len(self.y_val) > accuracy_treshold 
                                                       and roc_val > roc_treshold):
            y_hat_all = self.model.predict(self.alldata_x)
            y_hat_all_round = [round(x[1]) for x in y_hat_all]
            y_hat_all_confidence = [abs(x-y[1]) for (x,y) in zip(self.alldata_y, y_hat_all)]# чем меньше, тем больше уверенность модели
            #отбираем те 10% примеров, в которых модель больше всего уверена
            y_hat_all_confidence_percentile = [1 if x < np.percentile(y_hat_all_confidence, 0.20) 
                                           else 0 for x in y_hat_all_confidence]
            
            onehot_y_pred_correct = [] # список правильно предсказанных примеров
            for i,j in zip(y_hat_all_round, self.alldata_y):
                if i == j:
                    onehot_y_pred_correct.append(1)
                else:
                    onehot_y_pred_correct.append(0)
            #отберем лишь те, которые правильно предсказаны среди тех, в
            #которых модель уверена больше всего
            self.averageweights.append([1 if (x == y == 1) else 0 
                                         for (x,y) in zip(onehot_y_pred_correct,  y_hat_all_confidence_percentile)])
        return
    
df = pd.read_pickle(picture_path)
Y_set = df['average_grade']
X_set = df.drop(columns=[ 'average_grade'])
#X_set = X_set[['price_trend_mean', 'time', 'time_sequence', 'price_move',  'PVbyT']]
X_set = X_set.values.tolist()
X_set_formated = []
for i in X_set:
    vector = []
    for _ in i:
        subvector = ast.literal_eval(_)
        vector.append(subvector)
    X_set_formated.append(vector)  
Y_set = Y_set.values.tolist()
X_set = np.array([np.array(x) for x in X_set_formated])
Y_set = np.array([x for x in Y_set])
#X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=42)


grabage_and_gold_list = np.array([0]*len(X_set))
for i in range (iterations):
    matrix2 = []
    X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=i+300)
    #скопируем данные, чтобы не портить исходники
    X_set2 = X_set.copy()
    X_set2 = X_set2.reshape(len(X_set2),1000,7,1)#1000 7
    Y_set2 = Y_set
    
    class_weights = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)
    ###########################################################################
    X_train = X_train.reshape(len(X_train),1000,7,1)#1000 7
    X_test = X_test.reshape(len(X_test),1000,7,1)
    #one-hot encode target column
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2, dtype='float32')
    Y_test = tf.keras.utils.to_categorical(Y_test, num_classes=2, dtype='float32')
    Y_set_2 = tf.keras.utils.to_categorical(Y_set, num_classes=2, dtype='float32')
    ###########################################################################
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(50, kernel_size=(3, 7), activation='relu', input_shape=(1000,7,1)))#50 1 7
#    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,1), strides=None, padding='valid', data_format='channels_last'))
    model.add(tf.keras.layers.Conv2D(25, kernel_size=(3,1), activation='relu'))#25  30 1
    model.add(tf.keras.layers.Conv2D(5, kernel_size=(3,1), activation='relu'))# 5 60 1
#    model.add(BatchNormalization()) 
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.compile(optimizer= tf.train.AdamOptimizer(learning_rate=0.00005, beta1=0.9, beta2=0.99, epsilon=1e-08), loss='categorical_crossentropy', metrics=['accuracy'])#0.00003
    history2  = model.fit(X_train, Y_train, epochs=16,
                           validation_data=(X_test, Y_test),  
                           class_weight=class_weights, 
                           callbacks=[waste_callback(training_data=(X_train, Y_train),
                                                     validation_data=(X_test, Y_test), 
                                                     alldata = (X_set2, Y_set2), pred_list = matrix2)])# 
    if len(matrix2) != 0:
        grabage_and_gold_list += np.array(matrix2[0])    
    
    
print(sum(grabage_and_gold_list))
np.savetxt(waste_grades_path, grabage_and_gold_list, delimiter=",")
