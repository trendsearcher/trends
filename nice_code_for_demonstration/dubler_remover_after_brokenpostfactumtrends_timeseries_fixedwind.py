# -*- coding: utf-8 -*-
"""
script №4 in order of applying
That script takes output of breakthrough_patterns, tisk data after purifier
and del every sample, which future window, that serves to make a label for that particular sample, 
overlapping with neighbour sample window. Also problem of different number of features is solved.
result written in a form of 1dataframe of about 30 numbers and 2dataframe with
very long matrix (7*1000) per sample.

скрипт содержит закоментированную предподготовку новых фичей, которые
представляют собой взаимодействие T, P, V. Но пока они не дали никакого эффекта

остается нерешенная проблема в виде сверхдолгих свечек во время клирингов. 
оставил эту проблему напотом и просто добавил в качестве новой фичи временной вектор, 
чтобы нейронка сама разобралась в том, что там за паузы в данных в одни и те же 
моменты.
"""
import pandas as pd
import numpy as np
import ast
import math
###############################################################################
input_trend_data = '../trends/normal_trends1218.csv'
input_tick_data = '../../history/pureSBER1218.csv'
outputfile = '../../trends_out_dublers_new/trends_outofdublers_TPV_1218.csv'
###############################################################################
window_delimeters = 5 # размерность сетки деления изображения
forecast_win_length = 100 # расстояние между концом окна предсказания предыдущего паттерна и началом слудующего
past_window = 5000 # фиксированная длина истории
###############################################################################
    
df = pd.read_csv(input_trend_data, header= 0, error_bad_lines=False)
# удаляем дублеры
df=df.drop_duplicates(subset=['trend_start', 'trend_end'], keep='first')
df = df.sort_values(by=['trend_end'])
print('до удаления дублеров:')
print(df.info())
#удаление паттернов, окна в будущее которых накладываются
df = df.sort_values(by=['trend_end'])
df["old_trend_end"] = df["trend_end"].shift(1)
df['forecast_window'] = df["old_trend_end"] + forecast_win_length
df['nonoverlap'] = df['trend_end'] > df['forecast_window']
# все ли удалено за 1 итерацию
df = df.loc[(df.nonoverlap == True)]
if df.loc[(df.nonoverlap == False)].empty:
    print('df is out of twins')
else:
    print('need double check')    
df.drop(columns=['old_trend_end',  'forecast_window', 'nonoverlap'], inplace=True)
###############################################################################
'''['direction', 'trend_start', 'trend_end', 'importance', 'k', 'b', 
   line_touching_x','dispersion', 'trend_lenght', 'r_squared_of_trend', 
   'tops_coordinates','tops_height','tops_width','tops_HW_ratio','tops_count',
   'peaks_coordinates','peaks_height','peaks_width','peaks_HW_ratio','peaks_count', 
   'height_pic']'''
# паттерны разные и их надо привести к одной форме. Вычисляем различные характеристики
# из списков величин разной длины, а также считаем другие параметры (среднее,
# максимум, минимум, медиана, sum), чтобы
# в итоге получить табличные данные для лампового МЛ.
df2 = df.copy()

trend_touching_list_mean = []
trend_touching_list_std =[]
trend_lenght_high_ratio_list = []
trend_touching_list_median = []
trend_H_list = []
 
tops_height_list_std = []
tops_HW_ratio_list_std = []
tops_width_list_std = []
tops_height_list_sum = []
tops_height_list_max = []
tops_height_list_mean = []
tops_width_list_mean = []
tops_height_list_median = []
tops_width_list_median = []
tops_HW_ratio_list_mean = []
tops_HW_ratio_list_median = []

peaks_width_list_median = []
peaks_height_list_median = []
peaks_HW_ratio_list_median = []
peaks_width_list_std = []
peaks_height_list_std =[]
peaks_HW_ratio_list_std = []
peaks_height_list_sum = []
peaks_height_list_max = []
peaks_HW_ratio_list_mean = []
peaks_width_list_mean = []
peaks_height_list_mean = []

price_trend_mean_list = []
price_trend_max_list = []
price_trend_min_list = []
price_trend_move_list = []
time_sequence_list = []
volume_list = []
time_list = []

#производные
#PbyV_list = []
#PbyT_list = []
#VbyT_list = []
#PVbyT_list = []
#PTbyV_list = []

###############################################################################
mydateparser = lambda x: pd.datetime.strptime(x, '%H:%M:%S.%f')
colnames = ['<TIME>', '<VOLUME>', '<PRICE>']
tick_data = pd.read_csv(input_tick_data, sep = ',', names=colnames, header = 0, 
                        parse_dates=['<TIME>'], date_parser=mydateparser)
tick_data['<PRICE>'].fillna(method = 'ffill', inplace = True)
price_column = tick_data['<PRICE>'].tolist()
volume_column = tick_data['<VOLUME>'].tolist()
time_column = tick_data['<TIME>'].tolist()

# нормировка на внутренние показатели тренда (самоподобие)
for index, row in df.iterrows():
    price_trend_vector = []
    price_max_vector = []
    price_min_vector = []
    price_mean_vector = []
    price_move_vector = []
    volume_vector = []
    time_vector = []
    time_sequence = []
    price_vector = []
    
    direction = row['direction']
    height_pic = row['height_pic']
    trend_lenght =row['trend_lenght']
    trend_end = row['trend_end']
    trend_start = trend_end - past_window
    tang = row['k']
    b = row['b']
    a = list(range(trend_start,trend_end+1))[window_delimeters::window_delimeters]
    
    subtraction = 0
    if direction == 1:
        subtraction = 1
    else:
        subtraction = -1
    for i ,j in zip(a[:-1], a[1:]):
        P_dirty = price_column[i:j]
        Trend_P = [_*tang + b for _ in range (i,j)]
       
        P = subtraction*(np.array(Trend_P) - np.array(P_dirty))
        V = volume_column[i:j]
        T = time_column[i:j]
        if math.isnan(np.max(P)):
            price_max_vector_value = np.mean(price_max_vector)
        else:    
            price_max_vector_value = np.max(P)
        if math.isnan(np.min(P)):
            price_min_vector_value = np.mean(price_min_vector)
        else:    
            price_min_vector_value = np.min(P)
        if math.isnan(np.mean(P)):
            price_mean_vector_value = np.mean(price_mean_vector) 
        else:    
            price_mean_vector_value = np.mean(P)
        kkk = np.max(P_dirty) - np.min(P_dirty)    
        if math.isnan(kkk):
            price_move_vector_value = np.mean(price_move_vector)       
        else:    
            price_move_vector_value = kkk
        if np.sum(V) !=0:
            volume_vector_value = np.sum(V)
        else:
            volume_vector_value = 1
       
        
        price_max_vector.append(price_max_vector_value)
        price_min_vector.append(price_min_vector_value)
        price_mean_vector.append(price_mean_vector_value)
        price_move_vector.append(price_move_vector_value) # макс движение внутри свечи
        volume_vector.append(volume_vector_value)
        time_interval = T[-1] - T[0]
        timestamp_of_interval_middle = T[len(T)//2]
        time_of_interval_middle = int(60*timestamp_of_interval_middle.hour + 
                                      timestamp_of_interval_middle.minute)
        time_sequence.append(time_of_interval_middle)# вектор с временем каждой свечи (10.00, 10.05, ...18.45, 19.05)
        time_per_step = int((1000000*time_interval.seconds + 
                             time_interval.microseconds)/window_delimeters)
        
        if time_per_step > 36600000: #ночь внутри свечки!
            if len(time_vector)!= 0: # если есть предыдущая свеча, то приравняем к ней
                time_per_step = time_vector[-1]
            else: #если нет, то к 10000
                time_per_step = 10000
        elif time_per_step == 0:
            time_per_step = 1
        time_vector.append(time_per_step)
    
    #экстремальные значения для внутренней нормировки векторов  
    max_hight_of_peak_over_line = max(price_max_vector) 
    max_mean_of_peak_over_line = max(price_mean_vector)
    max_low_of_peak_over_line = max(price_min_vector)
    max_speed_of_deals_over_line = max(time_vector)#min
    max_volume_of_deals_over_line = max(volume_vector)
    max_price_move = max(price_move_vector)
    
    min_hight_of_peak_over_line = min(price_max_vector)  
    min_mean_of_peak_over_line = min(price_mean_vector)
    min_low_of_peak_over_line = min(price_min_vector)
    min_speed_of_deals_over_line = min(time_vector)
    min_volume_of_deals_over_line = min(volume_vector)
    min_price_move = min(price_move_vector)

    # проверка нормировочных величинна zero devision
    if max_hight_of_peak_over_line == 0:
        max_hight_of_peak_over_line = 1
    if max_speed_of_deals_over_line == 0:
        max_speed_of_deals_over_line = 1
    if max_volume_of_deals_over_line == 0:
        max_volume_of_deals_over_line = 1
    if max_price_move == 0:
        max_price_move = 1  
    #нормировка векторов на нормировочные величины
    price_mean_vector = [(x-min_mean_of_peak_over_line)/max_mean_of_peak_over_line 
                         for x in price_mean_vector]#x/max_hight_of_peak_over_line
    price_max_vector = [(x-min_hight_of_peak_over_line)/max_hight_of_peak_over_line 
                        for x in price_max_vector]#x/max_hight_of_peak_over_line
    price_min_vector = [(x-min_low_of_peak_over_line)/max_low_of_peak_over_line 
                        for x in price_min_vector]#x/max_hight_of_peak_over_line
    volume_vector = [(x-min_volume_of_deals_over_line)/max_volume_of_deals_over_line 
                     for x in volume_vector]#x/max_volume_of_deals_over_line
    time_vector = [(x-min_speed_of_deals_over_line)/max_speed_of_deals_over_line 
                   for x in time_vector]#max_speed_of_deals_over_line/x
    time_sequence = [(x-600)/830 for x in time_sequence]
    price_move_vector = [(x-min_price_move)/max_price_move for x in price_move_vector]#x/
    ######производные##########################################################
#    PbyV_max = max([p/(v) for p, v in zip(price_move_vector, volume_vector)])    
#    PbyT_max = max([p/(t+0.0001) for p, t in zip(price_move_vector, time_vector)])
#    VbyT_max = max([v/(t+0.0001) for v, t in zip(volume_vector, time_vector)])
#    PVbyT_max = max([p*v/(t+0.0001) for p, v, t in zip(price_move_vector, volume_vector, time_vector)])
#
#    PTbyV_max = max([p*t/v for p, t, v in zip(price_move_vector,time_vector, volume_vector)])
#    PbyV = [p/(v*PbyV_max) for p, v in zip(price_move_vector, volume_vector)]
#    PbyT = [p/((t+0.0001)*PbyT_max) for p, t in zip(price_move_vector, time_vector)]
#    VbyT = [v/((t+0.0001)*VbyT_max) for v, t in zip(volume_vector, time_vector)]
#    PVbyT = [(p*v/(t+0.0001))/PVbyT_max for p, v, t in zip(price_move_vector, volume_vector, time_vector)]
#    PTbyV = [p*t/(v*PTbyV_max) for p, t, v in zip(price_move_vector,time_vector, volume_vector)]
    #округлим для простоты
    price_mean_vector = list(map(lambda x:round(x, 6), price_mean_vector))
    price_max_vector = list(map(lambda x:round(x, 6), price_max_vector))
    price_min_vector = list(map(lambda x:round(x, 6), price_min_vector))
    volume_vector = list(map(lambda x:round(x, 6), volume_vector))
    time_vector = list(map(lambda x:round(x, 6), time_vector))
    price_move_vector = list(map(lambda x:round(x, 6), price_move_vector))
    #округлим производные
#    PbyV = list(map(lambda PbyV:round(PbyV, 6), PbyV))
#    PbyT = list(map(lambda PbyT:round(PbyT, 6), PbyT))
#    VbyT = list(map(lambda VbyT:round(VbyT, 6), VbyT))
#    PVbyT = list(map(lambda PVbyT:round(PVbyT, 6), PVbyT))
#    PTbyV = list(map(lambda PTbyV:round(PTbyV, 6), PTbyV))
    
    
    ##проверка картинок на Nan#################################################
    for x in price_mean_vector:
        if math.isnan(x):
            print('price_mean_vector nan')
    for x in price_max_vector:
        if math.isnan(x):
            print('price_max_vector nan')
    for x in price_min_vector:
        if math.isnan(x):
            print('price_min_vector nan')
    for x in price_move_vector:
        if math.isnan(x):
            print('price_move_vector nan')
    for x in time_sequence:
        if math.isnan(x):
            print('time_sequence nan')
    for x in volume_vector:
        if math.isnan(x):
            print('volume_vector nan') 
    for x in time_vector:
        if math.isnan(x):
            print('time_vector nan')        

    price_trend_mean_list.append(price_mean_vector)
    price_trend_max_list.append(price_max_vector)
    price_trend_min_list.append(price_min_vector)
    price_trend_move_list.append(price_move_vector)
    time_sequence_list.append(time_sequence)
    volume_list.append(volume_vector)
    time_list.append(time_vector)
    #производные
#    PbyV_list.append(PbyV)
#    PbyT_list.append(PbyT)
#    VbyT_list.append(VbyT)
#    PVbyT_list.append(PVbyT)
#    PTbyV_list.append(PTbyV)
    
    trend_lenght_high_ratio_list.append(height_pic/trend_lenght)
    line_touching_x = ast.literal_eval(row['line_touching_x'])[1:]#
    tops_height = ast.literal_eval(row['tops_height'])
    peaks_width = ast.literal_eval(row['peaks_width'])
    peaks_height = ast.literal_eval(row['peaks_width'])
    tops_width = ast.literal_eval(row['tops_width'])
    peaks_HW_ratio = ast.literal_eval(row['peaks_HW_ratio'])
    tops_HW_ratio = ast.literal_eval(row['tops_HW_ratio'])
    trend_H = height_pic
    tops_width_norm = [x/trend_lenght for x in tops_width]#/trend_lenght
    tops_height_norm = [abs(x)/trend_H for x in tops_height]#/height_pic
    peaks_width_norm = [x/trend_lenght for x in peaks_width]#/trend_lenght
    line_touching_x_norm = [(x - trend_start) /trend_lenght for x in line_touching_x]#[1:]
    peaks_height_norm = [abs(x)/trend_H for x in peaks_height]#/height_pic
    peaks_HW_ratio_norm = [abs(x)*trend_lenght/trend_H for x in peaks_HW_ratio]#*trend_lenght/height_pic
    tops_HW_ratio_norm = [abs(x) *trend_lenght / trend_H for x in tops_HW_ratio]#*trend_lenght/height_pic
    
    
    trend_H_list.append(abs(tang*trend_lenght))
    tops_height_list_sum.append(np.sum(tops_height_norm))
    tops_height_list_max.append(np.max(tops_height_norm))
    tops_height_list_std.append(np.std(tops_height_norm))
    tops_height_list_mean.append(np.mean(tops_height_norm))
    peaks_width_list_std.append(np.std(peaks_width_norm))
    peaks_width_list_mean.append(np.mean(peaks_width_norm))
    tops_width_list_std.append(np.std(tops_width_norm))
    tops_width_list_mean.append(np.mean(tops_width_norm))
    trend_touching_list_mean.append(np.mean(line_touching_x_norm))
    trend_touching_list_std.append(np.std(line_touching_x_norm))   
    peaks_height_list_std.append(np.std(peaks_height_norm))
    peaks_height_list_mean.append(np.mean(peaks_height_norm))
    peaks_height_list_sum.append(np.sum(peaks_height_norm))
    peaks_height_list_max.append(np.max(peaks_height_norm))
    peaks_HW_ratio_list_std.append(np.std(peaks_HW_ratio_norm))
    peaks_HW_ratio_list_mean.append(np.mean(peaks_HW_ratio_norm))
    tops_HW_ratio_list_std.append(np.std(tops_HW_ratio_norm))
    tops_HW_ratio_list_mean.append(np.mean(tops_HW_ratio_norm))
    
    trend_touching_list_median.append(np.median(line_touching_x_norm))
    tops_height_list_median.append(np.median(tops_height_norm))
    peaks_width_list_median.append(np.median(peaks_width_norm))
    tops_width_list_median.append(np.median(tops_width_norm))
    peaks_height_list_median .append(np.median(peaks_height_norm))
    peaks_HW_ratio_list_median .append(np.median(peaks_HW_ratio_norm))
    tops_HW_ratio_list_median .append(np.median(tops_HW_ratio_norm))
 
    
df2['trend_H'] = pd.Series(trend_H_list, index=df2.index)  
df2['trend_touching_std'] = pd.Series(trend_touching_list_std, index=df2.index)  
df2['trend_touching_mean'] = pd.Series(trend_touching_list_mean, index=df2.index)
df2['trend_touching_median'] = pd.Series(trend_touching_list_median, index=df2.index)
df2['tops_height_std'] = pd.Series(tops_height_list_std, index=df2.index) 
df2['tops_height_mean'] = pd.Series(tops_height_list_mean, index=df2.index) 
df2['tops_height_median'] = pd.Series(tops_height_list_median, index=df2.index) 
df2['tops_height_sum'] = pd.Series(tops_height_list_sum, index=df2.index) 
df2['tops_height_max'] = pd.Series(tops_height_list_max, index=df2.index) 
df2['peaks_width_std'] = pd.Series(peaks_width_list_std, index=df2.index) 
df2['peaks_width_mean'] = pd.Series(peaks_width_list_mean, index=df2.index) 
df2['peaks_width_median'] = pd.Series(peaks_width_list_median, index=df2.index) 
df2['tops_width_std'] = pd.Series(tops_width_list_std, index=df2.index) 
df2['tops_width_mean'] = pd.Series(tops_width_list_mean, index=df2.index) 
df2['tops_width_median'] = pd.Series(tops_width_list_median, index=df2.index) 
df2['peaks_height_std'] = pd.Series(peaks_height_list_std, index=df2.index) 
df2['peaks_height_mean'] = pd.Series(peaks_height_list_mean, index=df2.index) 
df2['peaks_height_median'] = pd.Series(peaks_height_list_median, index=df2.index) 
df2['peaks_height_sum'] = pd.Series(peaks_height_list_sum, index=df2.index) 
df2['peaks_height_max'] = pd.Series(peaks_height_list_max, index=df2.index) 
df2['tops_HW_ratio_std'] = pd.Series(tops_HW_ratio_list_std, index=df2.index) 
df2['tops_HW_ratio_mean'] = pd.Series(tops_HW_ratio_list_mean, index=df2.index) 
df2['tops_HW_ratio_median'] = pd.Series(tops_HW_ratio_list_median, index=df2.index) 
df2['peaks_HW_ratio_std'] = pd.Series(peaks_HW_ratio_list_std, index=df2.index) 
df2['peaks_HW_ratio_mean'] = pd.Series(peaks_HW_ratio_list_mean, index=df2.index) 
df2['peaks_HW_ratio_median'] = pd.Series(peaks_HW_ratio_list_median, index=df2.index) 
df2['trend_lenght_high_ratio'] = pd.Series(trend_lenght_high_ratio_list, index=df2.index) 


df2.drop(columns=['line_touching_x'], inplace=True)
df2.drop(columns=['peaks_coordinates'], inplace=True)
df2.drop(columns=['tops_coordinates'], inplace=True)
df2.drop(columns=['dispersion'], inplace=True)
df2.drop(columns=['tops_height'], inplace=True)
df2.drop(columns=['tops_width'], inplace=True)
df2.drop(columns=['peaks_width'], inplace=True)
df2.drop(columns=['peaks_height'], inplace=True)
df2.drop(columns=['peaks_HW_ratio'], inplace=True)
df2.drop(columns=['tops_HW_ratio'], inplace=True)
#df2.drop(columns=['trend_lenght'], inplace=True)
#df2.drop(columns=['height_pic'], inplace=True)


#датафрейм с картинками
df2['price_trend_mean'] = pd.Series(price_trend_mean_list, index=df2.index)  
df2['price_trend_max'] = pd.Series(price_trend_max_list, index=df2.index)  
df2['price_trend_min'] = pd.Series(price_trend_min_list, index=df2.index)  
df2['volume'] = pd.Series(volume_list, index=df2.index)  
df2['time'] = pd.Series(time_list, index=df2.index)
df2['time_sequence'] = pd.Series(time_sequence_list, index=df2.index)
df2['price_move'] = pd.Series(price_trend_move_list, index=df2.index)  
#производные
#df2['PbyV'] = pd.Series(PbyV_list, index=df2.index)  
#df2['PbyT'] = pd.Series(PbyT_list, index=df2.index)  
#df2['VbyT'] = pd.Series(VbyT_list, index=df2.index)  
#df2['PVbyT'] = pd.Series(PVbyT_list, index=df2.index)  
#df2['PTbyV'] = pd.Series(PTbyV_list, index=df2.index)  


print(df2.head(2))
print(df2.info())
df2.dropna(inplace=True)
df2.round(5) 
df2.to_csv(outputfile, index=False)  





















  
