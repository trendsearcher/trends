# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 10:41:01 2018

@author: user_PC
"""

import csv
import tensorflow as tf  # deep learning library. Tensors are just multi-dimensional arrays
import numpy as np
import pandas
###############################################################################
datapath='C:\\Users\\user_PC\\Desktop\\ugly\\SBRFpure.csv'# тута тики
trendspath ="C:\\Users\\user_PC\\Desktop\\good_bad\\marked_normal_trends.csv" # тута ненормированные
colnames = ['<price_eba>']
m = 1000 # от 1000 до 4000 линейная зависимость
counter_list = []
'данная функция переводит ординаты тиковых точек графика в ординаты в диапазоне (0, n)'
def correction(x, xx, xn):
    x_corrected = (x-xn)/(xx-xn)
    return(x_corrected)
'''                                                          header = 1!!!!'''
data = pandas.read_csv(datapath, names=colnames, sep = '\t', header = 1)
cols = data.columns
y_column = data[cols[0]]
y_list = list(y_column)
x_list = list(range(len(y_list)))
with open (trendspath , "r") as csvfile:
    reader = csv.reader(csvfile, delimiter =  ",")
    my_list0 = list(reader)
    my_list_chisto_chisto = []
    for i in my_list0:
        if len(i) == 3: # в некоторых примерах почему-то по 2 пары оценок. нафиг их
            direction_of_deal = int(i[0])
            basic_parameters = i[1].split('[[')[1].split(',')
            list_of_marks = i[2].split(',')[0][1:]
            my_exp_mark = int(i[2].split(',')[0][1:]) 
            tech_mark = int(i[2].split(',')[1][:-1])
            first_dot = int(basic_parameters[0])
            last_dot = int(basic_parameters[1])
            if first_dot - (last_dot - first_dot) > 0:
                angle = float(basic_parameters[3].split('([')[1])
                b_coeff = float(basic_parameters[4].split('])')[0])
                counter_list.append([direction_of_deal, first_dot, last_dot, angle, b_coeff, my_exp_mark])
my_list = sorted(counter_list, key = lambda  x: x[1]) # сортируем       
max_counter = len(my_list) - 5
k = 0
X_data = [] # картинки
Y_data = [] # оценки картинок
while k <= max_counter:
    line_y = []
    line_y_trend = []
    direction = my_list[k][0]
    mark = my_list[k][-1] - 1
    first_dot_0 = my_list[k][1]
    last_dot = my_list[k][2]
    len_x = last_dot - first_dot
    first_dot = first_dot_0 - int(len_x/2)
    angle = my_list[k][3]
    b_coeff = my_list[k][4]
    xplot = x_list[first_dot:last_dot] #список абсцисс картинки
    yplot = y_list[first_dot:last_dot] #список ординат картинки без вычета тренда
    yplot_trend = []#список ординат картинки с учетом вычетенного тренда
    if direction == 2:# линия поддержки. график - тренд
        for ii, jj in zip(xplot, yplot):
            corr_val = jj - angle*ii - b_coeff
            yplot_trend.append(corr_val)
    if direction == 1:# линия сопротивления. тренд - график
        for ii,jj in zip(xplot, yplot):
            corr_val = angle*ii + b_coeff - jj
            yplot_trend.append(corr_val)
    y_max = max(yplot_trend)  
    y_min = min(yplot_trend)                       
    step = int((len(xplot))/m)# ширина свечи
    step_list = list(range(m))
    for i,j in zip(step_list[:-1], step_list[1:]):
        candle_start = i*step
        candle_end = j*step
        candlex = xplot[candle_start:candle_end]
        candley = yplot_trend[candle_start:candle_end]
        candle_max = max(candley)
        candle_min = min(candley)
        candle_mean = np.mean(candley)
        candle_max_corrected = correction(candle_max, y_max, y_min)
        candle_min_corrected = correction(candle_min, y_max, y_min)
        candle_mean_corrected = correction(candle_mean, y_max, y_min) 
        candle_list = [candle_max_corrected, candle_min_corrected, candle_mean_corrected]
        line_y.append(candle_list)
    k += 1
    X_data.append(line_y)
    Y_data.append(mark)
X_data = np.array(X_data)
Y_data = np.array(Y_data)
###############################################################################
x_train = X_data[:100]
y_train = Y_data[:100]
x_test = X_data[100:]
y_test = Y_data[100:]


#mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
#(x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

#x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
#x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(4, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, epochs=30)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy
model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
prediction = new_model.predict([x_test])# дает распределение вероятностей
print(np.argmax(prediction(0)))



#import matplotlib.pyplot as plt
#plt.imshow(x_test[0], cmap = plt.cm.binary)
#plt.show
