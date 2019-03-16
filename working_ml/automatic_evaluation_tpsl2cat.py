# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
wind_parts = 50

 #рассчитывает зависимость квадрата стандартного отклонения Y от длины пути по оси X
def std_dependence_upon_x (Y_set) :
    ticks_total = len(Y_set) 
    WIN1 = int(ticks_total/100)
    WIN2 = int(WIN1/10)
    if WIN2 == 0:
        WIN2 = 1
    WIN3 =  int(WIN1/10)
    if WIN3 == 0:
        WIN3 = 1
    distrib_std = [] # standart deviation
    distrib_curr_window = []
    for window in range(1, WIN1, WIN2):
        window_i = []
        for l in range (1, (ticks_total - window), WIN3): 
            H = (price(l) - price(l + window))
            window_i.append(H)       
        distrib_std.append(np.std(window_i))
        distrib_curr_window.append(window)
    distrib_std1 = [x*x for x in distrib_std]
    A = np.vstack([distrib_curr_window, np.ones(len(distrib_curr_window))]).T
    m, c = np.linalg.lstsq(A, distrib_std1,rcond=-1)[0]
    return(m)
def price(some): # ввел функцию цены акции от тика, индекс начинается с единицы
    return (y_column[some])    
trend_data = pd.read_csv('C:\\Users\\user_PC\\Desktop\\rts\\normal_trends_outofdublers_norm.csv', header= 0, error_bad_lines=False)

cols = trend_data.columns
type_of_line = trend_data[cols[0]]# 1 - строю сверху, 2 - строю снизу
start = trend_data[cols[1]]
end = trend_data[cols[2]]
colnames = ['<TIME>', '<VOLUME>', '<PRICE>']
tick_data = pd.read_csv('C:\\Users\\user_PC\\Desktop\\rts\\pureRTS18.csv', sep = ',', names=colnames, header = 0)
y_column = tick_data['<PRICE>']#y_column = data[cols[0]]
y_column = list(y_column)

ticks_total = len(y_column) - 1 
x_list = list(range(len(y_column)))
list_of_coefficient_of_standart_deviation_before = []
list_of_coefficient_of_standart_deviation_after = []
grade_list = []

list_segmented = []
for i, j, direction in zip(start, end, type_of_line):
    start_price_of_walk_to_future = price(j)
    future_window = int((j - i)/20) #j - i
    future_window_end = j + future_window
    if future_window_end > ticks_total:
        future_window_end = ticks_total
    segment_of_future_window_list = [(x*int(future_window/wind_parts) + j)  for x in range(0,wind_parts)]# начинается с пересечения линии и содержит 11 чисел
    coefficient_of_standart_deviation1 = std_dependence_upon_x (y_column[i:j])
    list_of_coefficient_of_standart_deviation_before.append(coefficient_of_standart_deviation1)
    
    level_list = []
    counter = 0
    for kk in zip (y_column[j:future_window_end]):
        counter += 1
        if direction == 1:  # линия сопротивления
            level_list.append(float(kk[0]) - start_price_of_walk_to_future)
        if direction == 2:
            level_list.append(start_price_of_walk_to_future - float(kk[0]))
            
    
    tp = max(level_list)
    if tp < 0:
        tp = 0
    sl = abs(min(level_list))
    if sl == 0:
        sl = 1
    average_mark = tp/sl       
    if average_mark > 1:
        average_mark = 1
    else:
        average_mark = 0
    grade_list.append(average_mark)
    
print(np.mean(grade_list))    
trend_data['coeff_of_stand_deviat_before'] = pd.Series(list_of_coefficient_of_standart_deviation_before, index=trend_data.index)
random_rank = np.random.randint(2, size=len(grade_list))
trend_data['k'] = abs(trend_data['k'])

# данные нужны для формирования вектора P, V, T, но не нужны для обучения
trend_data.drop(columns=['b'], inplace=True)
trend_data.drop(columns=['trend_start'], inplace=True)
trend_data.drop(columns=['trend_end'], inplace=True)
trend_data.drop(columns=['direction'], inplace=True)
#
#
#
#

# генерация фич
#from sklearn  import preprocessing
#poly = preprocessing.PolynomialFeatures(3)
#trend_data2 = poly.fit_transform(trend_data)
#trend_data3 = pd.DataFrame(trend_data2, columns=poly.get_feature_names(trend_data.columns))
#trend_data3.drop(columns=['1'], inplace=True)


#нормировка строк друг относительно друга
def normalize(df):
    result = df.copy()
    for feature_name in df.columns: 
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return (result)

trend_data = normalize(trend_data)
trend_data['average_grade'] = pd.Series(grade_list, index=trend_data.index)#pd.Series(grade_list_segmented_aver, index=trend_data.index)
print(trend_data.columns)
trend_data.to_csv('C:\\Users\\user_PC\\Desktop\\rts\\normal_trends_outofdublers_norm_graded.csv', index=False)


















                    
         
       
    
