# -*- coding: utf-8 -*-
"""
Created on Sat May  4 20:59:38 2019

@author: user_PC
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

input_data = 'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_graded_all.csv'

trees = 200


def similarity(a, b):
    if len(a) != len(b):
        raise ValueError("Lists of different length.")
    return sum(i == j for i, j in zip(a, b))/len(a)
df = pd.read_csv(input_data, header= 0, error_bad_lines=False)
Y_set = df['average_grade']
X_set = df.drop(columns=[ 'average_grade'])

index_list_of_whole_dataset = list(range(len(Y_set)))

group_of_items = index_list_of_whole_dataset               # a sequence or set will work here.

acc_list = []
auc_list = []
for i in range(50):
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_set, Y_set, test_size=0.2, random_state=i)
    X_train = X_train.values.tolist()
    X_test = X_test.values.tolist()
    Y_train = Y_train.values.tolist()
    Y_test= Y_test.values.tolist()
    
    X_train = np.array([np.array(x) for x in X_train])
    X_test = np.array([np.array(x) for x in X_test])
    Y_train = np.array([np.array(x) for x in Y_train])
    Y_test = np.array([np.array(x) for x in Y_test])
    
    rnd_clf = RandomForestClassifier(n_estimators=trees, max_leaf_nodes=3, n_jobs=-1)
#    rnd_clf = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)

    rnd_clf.fit(X_train, Y_train)
    y_pred_rf = rnd_clf.predict(X_test)
    
    #roc = roc_auc_score(Y_test, y_pred_rf)
    roc_val = roc_auc_score(Y_test, y_pred_rf)
    acc_list.append(similarity(y_pred_rf, Y_test))
    auc_list.append(roc_val)
    
    
print('лес: %s' % trees)
print('accuracy_mean: %s' % np.mean(acc_list))
print('accuracy_std: %s' % np.std(acc_list))
print('бустинг')
print('roc-auc_val_mean: %s' % np.mean(auc_list))
print('roc-auc_val_std: %s' % np.std(auc_list))
    
    
    
    
    
    
    
    