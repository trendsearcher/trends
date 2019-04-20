# -*- coding: utf-8 -*-
"""
script №2 in order of applying
That script takes tick_data from purifier_zipifier_of_row_data, scans
it with window and writes every possible candidate to real trends in csv.file
minmax dots on plot are selected by hiperbolic distributed window grid 
(len of window is greater in far past, then in near past)
"""
import csv
import math
import numpy as np
import pandas
inputpath='../../trends_data/preprocess/pureSBER315.csv'
historyOutPath='../../trends_data/preprocess/shit.csv'


columns = ['<TIME>', '<VOLUME>', '<PRICE>']
###############################################################################
data = pandas.read_csv(inputpath, sep = ',', names=columns, header = 0)
y_column = data['<PRICE>']
y_list = np.array(y_column)
x_list = np.array(range(len(y_list)))
ticks_total = len(y_list) - 1          
#####################################
step_read = 60
Lmax = 400000
zazo_coeff = 0.00005
relax_coef = 1
number_of_bars = 800 # сетка на 800 делений
pitstop = ticks_total - 1000 # место остановки окна
min_trend_len = 3000
###############################################################################
breaker = 5000 # старт движения окна
###############################################################################
global x_stop
global direction
direction = 456# инициализация переменной не нулем и не единицей
x_stop = 0
#############################################################
def price(some): # ввел функцию цены акции от тика, индекс начинается с единицы
    return (y_list[some])
def func(breaker, zazor, historyfile):
    global start_short_triangle
    end_short_triangle = breaker
    if   end_short_triangle <= Lmax:
        start_short_triangle = 10
    elif  end_short_triangle > Lmax :
        start_short_triangle =  end_short_triangle - Lmax
    section_to_separate = end_short_triangle - start_short_triangle
    
    list_of_delimeters = []
    a = section_to_separate/math.log(number_of_bars)
    delimeter_position = start_short_triangle
    i = 1
    while i <= number_of_bars:
        delimeter_position += int(a/i)
        list_of_delimeters.append(delimeter_position)
        i += 1
    list_of_delimeters = sorted(list(set(list_of_delimeters)))
    list_of_mins = np.array(0)
    list_of_mins_pos = np.array(0)
    list_of_maxs = np.array(0) 
    list_of_maxs_pos = np.array(0)
    for i, j in zip(list_of_delimeters[:-1], list_of_delimeters[1:]):
        y_list_part = y_list[i:j]
        
        maxx = max(y_list_part)
        minn = min(y_list_part)
        list_of_maxs = np.append(list_of_maxs, maxx)
        list_of_mins = np.append(list_of_mins, minn)
        maxx_pos = i + np.argmax(y_list_part)
        minn_pos = i + np.argmin(y_list_part)
        list_of_maxs_pos = np.append(list_of_maxs_pos, maxx_pos)
        list_of_mins_pos = np.append(list_of_mins_pos, minn_pos)
    list_of_mins = np.delete(list_of_mins, 0)   
    list_of_mins_pos = np.delete(list_of_mins_pos, 0)   
    list_of_maxs = np.delete(list_of_maxs, 0)   
    list_of_maxs_pos = np.delete(list_of_maxs_pos, 0) 
    working_max =  list_of_maxs[-1]
    working_min = list_of_mins[-1]
    working_max_pos =  list_of_maxs_pos[-1]
    working_min_pos = list_of_mins_pos[-1]
    counter_max_list = list(range(len(list_of_maxs_pos)))
    ############################################################################
    # перебираем пары значений из списка максимумов
    for i, j, separ in  zip(list_of_maxs[:-1], list_of_maxs_pos[:-1], counter_max_list): 
        k = (working_max - i)/(working_max_pos - j)
        b = working_max - k*working_max_pos
        # временный список значений абсцыссы, в которых касательная касается графика в 4 и более точках
        list_kasanie_max = [] 
        # перебираем пары значений из списка максимумов
        for ii, jj in  zip(list_of_maxs[separ:-1], list_of_maxs_pos[separ:-1]): 
            if (ii - (k * jj + b)) > zazor:
                list_kasanie_max = []
            elif (k * jj + b - ii) > zazor:
                pass
            else:
                list_kasanie_max.append(jj)
        if len(list_kasanie_max) >= 4 and (list_kasanie_max[-1] - list_kasanie_max[0] > min_trend_len) :
            list_kasanie_max.append(working_max_pos)
            memorized_set.add((1, list_kasanie_max[0], list_kasanie_max[-1], k ,b, len(list_kasanie_max)))
#            csv.writer(historyfile).writerow([1, list_kasanie_max[0], list_kasanie_max[-1], k ,b, var, len(list_kasanie_max)])
    ############################################################################
    for i, j, separ in  zip(list_of_mins[:-1], list_of_mins_pos[:-1], counter_max_list): # перебираем пары значений из списка максимумов
        k = (working_min - i)/(working_min_pos - j)
        b = working_min - k*working_min_pos
        list_kasanie_min = [] # временный список значений абсцыссы, в которых касательная касается графика в 4 и более точках
        for ii, jj in  zip(list_of_mins[separ:-1], list_of_mins_pos[separ:-1]): # перебираем пары значений из списка максимумов
            if (k * jj + b - ii) > zazor:
                list_kasanie_min = []
            elif  (ii - (k * jj + b)) > zazor:
                pass
            else:
                list_kasanie_min.append(jj)
        if len(list_kasanie_min) >= 4 and (list_kasanie_min[-1] - list_kasanie_min[0] > min_trend_len):
            list_kasanie_min.append(working_min_pos)
            memorized_set.add((2, list_kasanie_min[0], list_kasanie_min[-1], k ,b,len(list_kasanie_min)))
#            csv.writer(historyfile).writerow([2, list_kasanie_min[0], list_kasanie_min[-1], k ,b,len(list_kasanie_min)])
    return
###############################################################################
historyfile = open(historyOutPath, 'a',newline='')
memorized_set = set()
try:
    while breaker <= pitstop:
        zazor = zazo_coeff*price(breaker)
        func1_result = func(breaker, zazor, historyfile)
        breaker += step_read
        print(breaker)
except:        
    memorized_list = list (memorized_set) 
    memorized_list = [list(x) for x in memorized_list] 
    memorized_list.sort(key=lambda x: x[1])
    #######################вводим заголовок########################################
    historyfile = open(historyOutPath,'a',newline='')
    writer = csv.writer(historyfile, delimiter=',')
    writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
    writer.writerows(memorized_list)
    historyfile.close()    
memorized_list = list (memorized_set) 
memorized_list = [list(x) for x in memorized_list] 
memorized_list.sort(key=lambda x: x[1])
#######################вводим заголовок########################################
historyfile = open(historyOutPath,'a',newline='')
writer = csv.writer(historyfile, delimiter=',')
writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
writer.writerows(memorized_list)
historyfile.close()    