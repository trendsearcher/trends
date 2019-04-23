# -*- coding: utf-8 -*-
"""
script №5 in order of applying 
That script takes output of dubler_remover_after_brokenpostfactumtrends_timeseries and
tisk data after purifier_zipifier_of_row_data and produce grade [1, 0] depending on 
behavior of price after breaking the trend. Result is written in form of 2 dataframes:
(numbers + grades, long matrices + grades)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
###############################################################################
input_trend_data = 'C:\\Users\\user_PC\\Desktop\\sber2\\normal_trends_outofdublers_norm_TPV_vectors615.csv'
input_tick_data = 'C:\\Users\\user_PC\\Desktop\\sber2\\pureSBER615.csv'

output_numbers = 'C:\\Users\\user_PC\\Desktop\\sber2\\normal_trends_outofdublers_norm_graded_minmax.csv'
output_pic = 'C:\\Users\\user_PC\\Desktop\\sber2\\normal_trends_outofdublers_norm_TPV_vectors_graded_minmax.pkl'

 #рассчитывает зависимость квадрата стандартного отклонения Y от длины пути по оси X
def std_dependence_upon_x (Y_set) :
    ticks_total = len(Y_set) 
    WIN1 = int(ticks_total/100)#100
    WIN2 = int(WIN1/10)#10
    if WIN2 == 0:
        WIN2 = 1
    WIN3 =  int(WIN1/10)#10
    if WIN3 == 0:
        WIN3 = 1
    distrib_std = [] # standart deviation
    distrib_curr_window = []
    for window in range(1, WIN1, WIN2):
        window_i = []
        for l in range (1, (ticks_total - window), WIN3): 
            H = (price(l) - price(l + window))
            if math.isnan(H):
                H = 0
            window_i.append(H)       
        distrib_std.append(np.std(window_i))
        distrib_curr_window.append(window)
    distrib_std1 = [x*x for x in distrib_std]
#    lines = plt.plot(distrib_curr_window, distrib_std1)
#    l1= lines
#    plt.setp(lines, linestyle='-')
#    plt.setp(l1, linewidth=1, color='b')    
#    plt.title('апрапр')
#    plt.grid()
#    plt.show()
#    plt.pause(0.05)
    A = np.vstack([distrib_curr_window[:3], np.ones(3)]).T
    m, c = np.linalg.lstsq(A, distrib_std1[:3],rcond=-1)[0]
    return(m)
def price(some): # ввел функцию цены акции от тика, индекс начинается с единицы
    return (y_column[some])    
trend_data = pd.read_csv(input_trend_data, header= 0, error_bad_lines=False)



cols = trend_data.columns
type_of_line = trend_data[cols[0]]# 1 - строю сверху, 2 - строю снизу
start = trend_data[cols[1]]
end = trend_data[cols[2]]

colnames = ['<TIME>', '<VOLUME>', '<PRICE>']
tick_data = pd.read_csv(input_tick_data, sep = ',', names=colnames, header = 0)
tick_data['<PRICE>'].fillna(method = 'ffill', inplace = True)
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
    future_window =int((j - i)/20) 
    future_window_end = j + future_window
    if future_window_end > ticks_total:
        future_window_end = ticks_total
    coefficient_of_standart_deviation1 = std_dependence_upon_x (y_column[i:j])
    list_of_coefficient_of_standart_deviation_before.append(coefficient_of_standart_deviation1)
    upper_boarder_list = []
    bottom_boarder_list = []
#    surface = 0
#    counter = 0
#    for kk in zip (y_column[j:future_window_end]):
#        counter += 1
#        if direction == 1:  # линия сопротивления
#            surface += (float(kk[0]) - start_price_of_walk_to_future)
#        if direction == 2:
#            surface += (start_price_of_walk_to_future - float(kk[0]))
#            
#    average_surface = surface/counter
#    if average_surface > 0:
#        average_surface = 1
#    else:
#        average_surface = 0
#    grade_list.append(average_surface)

    price_walk = []
    for kk in zip (y_column[j:future_window_end]):
        if direction == 1:  # линия сопротивления
            price_walk.append(float(kk[0]) - start_price_of_walk_to_future)
        if direction == 2:
            price_walk.append(start_price_of_walk_to_future - float(kk[0]))
    if min(price_walk) == 0:# чтобы не было деления на ноль
        mn = 0.01
    else:
        mn = abs(min(price_walk))   
    mx = max(price_walk)    
    average_surface = mx/mn
    if (average_surface) > 0:
        average_surface = 1
    else:
        average_surface = 0
    grade_list.append(average_surface)
    
print(np.mean(grade_list))    
trend_data['coeff_of_stand_deviat_before'] = pd.Series(list_of_coefficient_of_standart_deviation_before, index=trend_data.index)
#random_rank = np.random.randint(2, size=len(grade_list))
trend_data['k'] = abs(trend_data['k'])
# данные нужны для формирования вектора P, V, T, но не нужны для обучения
trend_data.drop(columns=['b'], inplace=True)
trend_data.drop(columns=['trend_start'], inplace=True)
trend_data.drop(columns=['trend_end'], inplace=True)
trend_data.drop(columns=['direction'], inplace=True)
#


#нормировка строк друг относительно друга
def normalize(df):
    result = df.copy()
    for feature_name in ['importance', 'k', 'trend_lenght', 'r_squared_of_trend', 'tops_count',
       'peaks_count', 'height_pic', 'trend_H', 'trend_touching_std',
       'trend_touching_mean', 'trend_touching_median', 'tops_height_std',
       'tops_height_mean', 'tops_height_median', 'tops_height_sum',
       'tops_height_max', 'peaks_width_std', 'peaks_width_mean',
       'peaks_width_median', 'tops_width_std', 'tops_width_mean',
       'tops_width_median', 'peaks_height_std', 'peaks_height_mean',
       'peaks_height_median', 'peaks_height_sum', 'peaks_height_max',
       'tops_HW_ratio_std', 'tops_HW_ratio_mean', 'tops_HW_ratio_median',
       'peaks_HW_ratio_std', 'peaks_HW_ratio_mean', 'peaks_HW_ratio_median',
       'trend_lenght_high_ratio', 'coeff_of_stand_deviat_before']: 
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return (result)

trend_data = normalize(trend_data)
trend_data['average_grade'] = pd.Series(grade_list, index=trend_data.index)#pd.Series(grade_list_segmented_aver, index=trend_data.index)
trend_data.dropna(inplace=True)
print(np.mean(trend_data[['average_grade']])) 
trend_data_numbers = trend_data[[ 'tops_count',
       'peaks_count', 'trend_touching_std',
       'trend_touching_mean', 'trend_touching_median', 'tops_height_std',
       'tops_height_mean', 'tops_height_median', 'tops_height_sum',
       'tops_height_max', 'peaks_width_std', 'peaks_width_mean',
       'peaks_width_median', 'tops_width_std', 'tops_width_mean',
       'tops_width_median', 'peaks_height_std', 'peaks_height_mean',
       'peaks_height_median', 'peaks_height_sum', 'peaks_height_max',
       'tops_HW_ratio_std', 'tops_HW_ratio_mean', 'tops_HW_ratio_median',
       'peaks_HW_ratio_std', 'peaks_HW_ratio_mean', 'peaks_HW_ratio_median',
       'trend_lenght_high_ratio',  'average_grade']]#'importance', 'k', 'trend_lenght', 'r_squared_of_trend', 'tops_count','peaks_count', 'height_pic', 'trend_H', 'trend_touching_std','trend_touching_mean', 'trend_touching_median', 'tops_height_std', 'tops_height_mean', 'tops_height_median', 'tops_height_sum', 'tops_height_max', 'peaks_width_std', 'peaks_width_mean','peaks_width_median', 'tops_width_std', 'tops_width_mean', 'tops_width_median', 'peaks_height_std', 'peaks_height_mean', 'peaks_height_median', 'peaks_height_sum', 'peaks_height_max','tops_HW_ratio_std', 'tops_HW_ratio_mean', 'tops_HW_ratio_median',  'peaks_HW_ratio_std', 'peaks_HW_ratio_mean', 'peaks_HW_ratio_median',   'trend_lenght_high_ratio', 'coeff_of_stand_deviat_before', 'average_grade'
trend_data_pic = trend_data[['price_trend_mean', 'price_trend_max','price_trend_min', 'volume', 'time', 'time_sequence','price_move', 'average_grade']]#,'PbyV', 'PbyT','VbyT', 'PVbyT', 'PTbyV']]
print(trend_data_pic.info())
print()
print(trend_data_numbers.info())
trend_data_numbers.to_csv(output_numbers, index=False)
trend_data_pic.to_pickle(output_pic)
#trend_data_pic.to_csv('C:\\Users\\user_PC\\Desktop\\sber2\\normal_trends_outofdublers_norm_TPV_vectors_graded615.csv')















                    
         
       
    
