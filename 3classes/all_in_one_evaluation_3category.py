# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:23:19 2019

@author: user_PC
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from scipy.stats import boxcox
history_directory = 'E:\\history\\'
trends_directory = 'E:\\trends_out_dublers_10\\'
graded_trends_numbers = 'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_graded_all_3class_20.csv'
graded_trends_pic = 'C:\\Users\\user_PC\\Desktop\\graded_trends_all\\normal_trends_outofdublers_norm_TPV_vectors_graded_all_3class_20.pkl'
proportion_of_standart_deviation = 0.001#коэффициент границы в выставлении оценки 005дает80Auc
min_boarder = 0 # минимальный баоьер на старте оцениваия
window_in_future = 20
colnames = ['<TIME>', '<VOLUME>', '<PRICE>']

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
    A = np.vstack([distrib_curr_window[:3], np.ones(3)]).T
    m, c = np.linalg.lstsq(A, distrib_std1[:3],rcond=-1)[0]
    return(m)
def price(some): # ввел функцию цены акции от тика
    return (y_column[some])    
frames = []# сюда собираю все датафреймы от разных контрактов
grade_list = [] # категория тренда
delta_list = [] # дельта, кторую даст стратегия на этом тренде
sorted_directory_list = ['trends_outofdublers_TPV_315.csv', 'trends_outofdublers_TPV_615.csv',
                         'trends_outofdublers_TPV_915.csv', 'trends_outofdublers_TPV_1215.csv', 
                         'trends_outofdublers_TPV_316.csv', 'trends_outofdublers_TPV_616.csv', 
                         'trends_outofdublers_TPV_916.csv', 'trends_outofdublers_TPV_1216.csv', 
                         'trends_outofdublers_TPV_317.csv', 'trends_outofdublers_TPV_617.csv', 
                         'trends_outofdublers_TPV_917.csv', 'trends_outofdublers_TPV_1217.csv', 
                         'trends_outofdublers_TPV_318.csv', 'trends_outofdublers_TPV_618.csv', 
                         'trends_outofdublers_TPV_918.csv', 'trends_outofdublers_TPV_1218.csv']
for trend_file in sorted_directory_list:
    print(trend_file)
    trend_data = pd.read_csv(trends_directory+trend_file, header= 0, error_bad_lines=False)
    cols = trend_data.columns
    type_of_line = trend_data[cols[0]]# 1 - строю сверху, 2 - строю снизу
    start = trend_data[cols[1]]
    end = trend_data[cols[2]]
    corresponding_tick_file_name = history_directory + 'pureSBER' + \
                                   trend_file.split('trends_outofdublers_TPV_')[1]
    tick_data = pd.read_csv(corresponding_tick_file_name, sep = ',',  \
                            names=colnames, header = 0)
    tick_data['<PRICE>'].fillna(method = 'ffill', inplace = True)
    y_column = tick_data['<PRICE>']#y_column = data[cols[0]]
    y_column = list(y_column)
    ticks_total = len(y_column) - 1 
    x_list = list(range(len(y_column)))
    list_of_coefficient_of_standart_deviation_before = []
    
    list_segmented = []
    for i, j, direction in zip(start, end, type_of_line):
#        start_price_of_walk_to_future = price(j)
#        future_window =int((j - i)/20) 
#        future_window_end = j + future_window
#        if future_window_end > ticks_total:
#            future_window_end = ticks_total
#        coefficient_of_standart_deviation1 = std_dependence_upon_x (y_column[i:j])
#        list_of_coefficient_of_standart_deviation_before.append(coefficient_of_standart_deviation1)
#        upper_boarder_list = []
#        bottom_boarder_list = []
 

        start_price_of_walk_to_future = price(j)
        future_window = int((j - i)/window_in_future) #j - i
        future_window_end = j + future_window
        if future_window_end > ticks_total:
            future_window_end = ticks_total
        coefficient_of_standart_deviation1 = std_dependence_upon_x (y_column[i:j])
        list_of_coefficient_of_standart_deviation_before.append(coefficient_of_standart_deviation1)

        upper_boarder_list = []
        bottom_boarder_list = []
        for k in range(j+int(future_window/1.5), future_window_end):
            if proportion_of_standart_deviation*math.sqrt(coefficient_of_standart_deviation1*k)  < min_boarder:
                upper_boarder = start_price_of_walk_to_future + min_boarder
            else:
                upper_boarder = start_price_of_walk_to_future + math.sqrt(coefficient_of_standart_deviation1*k*proportion_of_standart_deviation)
            if proportion_of_standart_deviation*math.sqrt(coefficient_of_standart_deviation1*k)  < min_boarder:
                bottom_boarder = start_price_of_walk_to_future - min_boarder
            else:
                bottom_boarder = start_price_of_walk_to_future - math.sqrt(coefficient_of_standart_deviation1*k*proportion_of_standart_deviation)    
            upper_boarder_list.append(upper_boarder)   
            bottom_boarder_list.append(bottom_boarder) 
        category_list = [] # 0 1 2 - тип тренда 
        delta_price = [] # запишем сюда дельту цены по абсолюту для 1 и 2 класса
        counter = 0
        for kk, up, down in zip (y_column[j+int(future_window/1.5):future_window_end], upper_boarder_list, bottom_boarder_list):
            counter += 1
            if direction == 1:  # линия сопротивления
                if kk - up > 0:
                    category_list.append(1)
                    delta_price.append((kk - up)/kk)
                elif  down - kk > 0:
                    category_list.append(2)#-1
                    delta_price.append((down - kk)/kk)
                else:
                    category_list.append(0)
                    delta_price.append(0)
            if direction == 2:
                if kk - up > 0:
                    category_list.append(2)#-1
                    delta_price.append((kk - up)/kk)
                elif down - kk > 0:
                    category_list.append(1)
                    delta_price.append((down - kk)/kk)
                else:
                    category_list.append(0)
                    delta_price.append(0)
        #при выходе в любую сторону из области случайных колебаний вне зависимости от дальнейшего поведения присваиваем оенку            
        try:
            the_grade = next((x for  x in (category_list) if x))
            the_delta = next((y for  x, y in zip(category_list,delta_price) if x))
        except:
            the_grade = 0 
            the_delta = 0
        grade_list.append(the_grade)  
        delta_list.append(the_delta)          
                    
##        surface = 0
##        counter = 0
##        for kk in zip (y_column[j:future_window_end]):
##            counter += 1
##            if direction == 1:  # линия сопротивления
##                surface += (float(kk[0]) - start_price_of_walk_to_future)
##            if direction == 2:
##                surface += (start_price_of_walk_to_future - float(kk[0]))
##                
##        average_surface = surface/counter
##        if average_surface > 0:
##            average_surface = 1
##        else:
##            average_surface = 0
##        grade_list.append(average_surface)
#    
#        price_walk = []
#        for kk in zip (y_column[j:future_window_end]):
#            if direction == 1:  # линия сопротивления
#                price_walk.append(float(kk[0]) - start_price_of_walk_to_future)
#            if direction == 2:
#                price_walk.append(start_price_of_walk_to_future - float(kk[0]))
#        if min(price_walk) == 0:
#            mn = 0.01
#        else:
#            mn = abs(min(price_walk))   
#        mx = max(price_walk)    
#        average_surface = mx/mn
#        if (average_surface) > 1:
#            average_surface = 1
#        else:
#            average_surface = 0
#        grade_list.append(average_surface)
        
    trend_data['coeff_of_stand_deviat_before'] = pd.Series(list_of_coefficient_of_standart_deviation_before, index=trend_data.index)
    trend_data['k'] = abs(trend_data['k'])
    # данные нужны для формирования вектора P, V, T, но не нужны для обучения
    trend_data.drop(columns=['b'], inplace=True)
    trend_data.drop(columns=['trend_start'], inplace=True)
    trend_data.drop(columns=['trend_end'], inplace=True)
    trend_data.drop(columns=['direction'], inplace=True)
    frames.append(trend_data)
trend_data_concated = pd.concat(frames)


#def fetch_waste_lablings():
#    '''достаю приготовленные оценки мусорности, складываю поэлементно и 
#    масштабирую от 0 до 1'''
#    list4 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating11.csv", delimiter=',')
#    list5 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating12.csv", delimiter=',')
#    list6 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating13.csv", delimiter=',')
#    list7 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating14.csv", delimiter=',')
#    list8 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating15.csv", delimiter=',')
#    list9 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating16.csv", delimiter=',')
#    list10 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating17.csv", delimiter=',')
#    list11 = np.genfromtxt("C:\\Users\\user_PC\\Desktop\\graded_trends_all\\samples_rating18.csv", delimiter=',')
#    res = (list4 + list5 +list6 +list7 +list8 +list9 +list10 +list11)
##    res = [(x-min(res))/max(res) for x in res]
#    res = [0 if x < np.mean(res) else 1 for x in res ]
#    return(res)

def normalize(df):
    '''нормировка строк друг относительно друга'''
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
#        df[feature_name] = boxcox(df[feature_name], lmbda=0.0)
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return (df)
normalized_df = normalize(trend_data_concated)
    
    
normalized_df['money'] = pd.Series(delta_list, index=normalized_df.index)#pd.Series(grade_list_segmented_aver, index=trend_data.index)
normalized_df['average_grade'] = pd.Series(grade_list, index=normalized_df.index)#pd.Series(grade_list_segmented_aver, index=trend_data.index)
normalized_df.dropna(inplace=True)
#normalized_df['wasteness'] = pd.Series(fetch_waste_lablings(), index=normalized_df.index)#pd.Series(grade_list_segmented_aver, index=trend_data.index)
a = normalized_df.loc[normalized_df['money'] != 0]
b = np.mean(a[['money']])
print('средние деньги в долях от контракта')
print(b)
print('доля вышедших из-за границы трендов')
print((a.shape[0])/(normalized_df.shape[0]))
#print(np.mean(normalized_df[['wasteness']])) 
trend_data_numbers = normalized_df[[ 'tops_count',
       'peaks_count', 'trend_touching_std',
       'trend_touching_mean', 'trend_touching_median', 'tops_height_std',
       'tops_height_mean', 'tops_height_median', 'tops_height_sum',
       'tops_height_max', 'peaks_width_std', 'peaks_width_mean',
       'peaks_width_median', 'tops_width_std', 'tops_width_mean',
       'tops_width_median', 'peaks_height_std', 'peaks_height_mean',
       'peaks_height_median', 'peaks_height_sum', 'peaks_height_max',
       'tops_HW_ratio_std', 'tops_HW_ratio_mean', 'tops_HW_ratio_median',
       'peaks_HW_ratio_std', 'peaks_HW_ratio_mean', 'peaks_HW_ratio_median',
       'trend_lenght_high_ratio',  'average_grade' , 'money']]#'importance', 'k', 'trend_lenght', 'r_squared_of_trend', 'tops_count','peaks_count', 'height_pic', 'trend_H', 'trend_touching_std','trend_touching_mean', 'trend_touching_median', 'tops_height_std', 'tops_height_mean', 'tops_height_median', 'tops_height_sum', 'tops_height_max', 'peaks_width_std', 'peaks_width_mean','peaks_width_median', 'tops_width_std', 'tops_width_mean', 'tops_width_median', 'peaks_height_std', 'peaks_height_mean', 'peaks_height_median', 'peaks_height_sum', 'peaks_height_max','tops_HW_ratio_std', 'tops_HW_ratio_mean', 'tops_HW_ratio_median',  'peaks_HW_ratio_std', 'peaks_HW_ratio_mean', 'peaks_HW_ratio_median',   'trend_lenght_high_ratio', 'coeff_of_stand_deviat_before', 'average_grade'
trend_data_pic = normalized_df[['price_trend_mean', 'price_trend_max','price_trend_min', 'volume', 'time', 'time_sequence','price_move', 'average_grade', 'money']]#,'PbyV', 'PbyT','VbyT', 'PVbyT', 'PTbyV']]
print(trend_data_pic.info())
print(trend_data_numbers.info())

trend_data_numbers.to_csv(graded_trends_numbers)
trend_data_pic.to_pickle(graded_trends_pic)
#
