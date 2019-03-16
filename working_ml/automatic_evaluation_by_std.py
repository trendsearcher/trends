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
proportion_of_standart_deviation_list = [0]#, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
trend_data = pd.read_csv('C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm.csv', header= 0, error_bad_lines=False)

cols = trend_data.columns
type_of_line = trend_data[cols[0]]# 1 - строю сверху, 2 - строю снизу
start = trend_data[cols[1]]
end = trend_data[cols[2]]

tick_data = pd.read_csv('C:\\Users\\user_PC\\Desktop\\sber\\SBER.csv',  sep = '\t', header = 1)
tick_cols = tick_data.columns
y_column = tick_data[tick_cols[0]]
y_column = list(y_column)
ticks_total = len(y_column) - 1 
x_list = list(range(len(y_column)))
list_of_coefficient_of_standart_deviation_before = []
list_of_coefficient_of_standart_deviation_after = []
grade_list_segmented_aver = []
for proportion_of_standart_deviation in proportion_of_standart_deviation_list:

    list_segmented = []
    for i, j, direction in zip(start, end, type_of_line):
        start_price_of_walk_to_future = price(j)
        future_window = int((j - i)/5) #j - i
        future_window_end = j + future_window
        if future_window_end > ticks_total:
            future_window_end = ticks_total
        segment_of_future_window_list = [(x*int(future_window/wind_parts) + j)  for x in range(0,wind_parts)]# начинается с пересечения линии и содержит 11 чисел
        coefficient_of_standart_deviation1 = std_dependence_upon_x (y_column[i:j])
        coefficient_of_standart_deviation2 = std_dependence_upon_x (y_column[j:future_window_end])
        list_of_coefficient_of_standart_deviation_before.append(coefficient_of_standart_deviation1)
        list_of_coefficient_of_standart_deviation_after.append(coefficient_of_standart_deviation2)
        upper_boarder_list = []
        bottom_boarder_list = []
        for k in range(j, future_window_end):
            min_boarder = 0
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
        grade_list = []
        counter = 0
        for kk, up, down in zip (y_column[j:future_window_end], upper_boarder_list, bottom_boarder_list):
            counter += 1
            if direction == 1:  # линия сопротивления
                if kk - up > 0:
                    grade_list.append(1)
                elif  down - kk > 0:
                    grade_list.append(-1)#-1
                else:
                    grade_list.append(0)
            if direction == 2:
                if kk - up > 0:
                    grade_list.append(-1)#-1
                elif down - kk > 0:
                    grade_list.append(1)
                else:
                    grade_list.append(0)
        # независимые оценки  с шагом в 10% от длины тренда от момента пересечения          
        grade_list_segmented = [0] * wind_parts
        _ = 0
        while _ <= len(segment_of_future_window_list) - 2:
            boarder1 = segment_of_future_window_list[_]
            boarder2 = segment_of_future_window_list[_ + 1]
            for grade, x  in zip(grade_list, x_list[j:future_window_end]):
                if x < boarder2 and x >=boarder1:
                    if grade != 0:
                        grade_list_segmented[_] = grade
                        break
            _ += 1        
        list_segmented.append([grade_list_segmented])
        simple_grade = np.mean([grade_list_segmented])
        if simple_grade > 0:
            grade_list_segmented_aver.append(1)
        else:
            grade_list_segmented_aver.append(0)
#    name_grade_column = str(proportion_of_standart_deviation)  
#    sLength = len(trend_data['trend_start'])
#    trend_data[('grade_list_' + name_grade_column)] = pd.Series(list_segmented, index=trend_data.index)
print(np.mean(grade_list_segmented_aver))
trend_data['coeff_of_stand_deviat_before'] = pd.Series(list_of_coefficient_of_standart_deviation_before, index=trend_data.index)
trend_data['coeff_of_stand_deviat_after'] = pd.Series(list_of_coefficient_of_standart_deviation_after, index=trend_data.index)
random_rank = np.random.randint(2, size=len(grade_list_segmented_aver))
trend_data['average_grade'] = pd.Series(grade_list_segmented_aver, index=trend_data.index)#pd.Series(grade_list_segmented_aver, index=trend_data.index)
trend_data['coeff_of_stand_deviat_after_norm'] = trend_data['coeff_of_stand_deviat_after']/trend_data["coeff_of_stand_deviat_before"]
trend_data.drop(columns=['coeff_of_stand_deviat_after'], inplace=True)
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
#poly = preprocessing.PolynomialFeatures(1)
#trend_data2 = poly.fit_transform(trend_data)
#trend_data3 = pd.DataFrame(trend_data2, columns=poly.get_feature_names(trend_data.columns))
#trend_data3.drop(columns=['1'], inplace=True)



#нормировка строк друг относительно друга
def normalize(df):
    result = df.copy()
    for feature_name in df.columns[:-2]: 
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return (result)

trend_data = normalize(trend_data)
trend_data.to_csv('C:\\Users\\user_PC\\Desktop\\sber\\normal_trends_outofdublers_norm_graded.csv', index=False)


















                    
         
       
    
