# -*- coding: utf-8 -*-
"""
script №2 in order of applying
That script takes data from purifier, scans
it with window and writes every possible candidate to real trends in csv.file
(candidates may seem like dublers, but they are not)
minmax dots on plot are selected by hiperbolic distributed window grid 
(len of window is greater in far past, then in near past)
"""
import csv
import math
import numpy as np
import pandas

input_file_name = '../sber/pureSBER19.csv'
output_pure_file_name = '../sber/raw_signal.csv'
columns = ['<TIME>', '<VOLUME>', '<PRICE>']
###############################################################################
data = pandas.read_csv(input_file_name, sep = ',', names=columns,  header = 0)
y_column = data['<PRICE>']
y_list = np.array(y_column)
x_list = np.array(range(len(y_list)))
ticks_total = len(y_list) - 1 

step_read = 60 # шаг обработки
Lmax = 400000 # макс длина паттерна
zazo_coeff = 0.00005
number_of_bars = 800 # сетка на 800 делений
pitstop = ticks_total - 1000 # место остановки обхода истории
min_trend_len = 3000 # мин длина паттерна
start_walking = 5000 # старт движения окна 5000
###############################################################################

def price(some): 
    'ценa акции от тика'
    return (y_list[some])

# Собираем в сет паттерны
memorized_set = set()
###############################################################################
def _func(start_walking, zazor, historyfile):
    '''ничего не возвращаюшая функция (кроме эксепшн), но код было удобно 
    завернуть в func.
    служит для заполнения паттернами сета, инициализированного снаружи.
    на переданном в функцию файле с историей строится сетка от старта в 
    прошлое, где шаги сетки имеют гиперболическое распределение (чем дальше в
    прошлое, тем боьше шаг). на каждой ячейке по экстремумам функци пытается 
    уложить паттерн с точностью в zazor, переданный в функцию.
    если набирается минимальное количество контактов паттерна с графиком,
    то такой паттерн записывается в сет.'''
    end_short_triangle = start_walking
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
    list_of_mins = []
    list_of_mins_pos = []
    list_of_maxs = [] 
    list_of_maxs_pos = []
    for i, j in zip(list_of_delimeters[:-1], list_of_delimeters[1:]):
        y_list_part = y_list[i:j]
        list_of_maxs.append(max(y_list_part))
        list_of_mins.append(min(y_list_part))
        maxx_pos = i + np.argmax(y_list_part)
        minn_pos = i + np.argmin(y_list_part)
        list_of_maxs_pos.append(maxx_pos)
        list_of_mins_pos.append(minn_pos)
    working_max =  list_of_maxs[-1]
    working_min = list_of_mins[-1]
    working_max_pos =  list_of_maxs_pos[-1]
    working_min_pos = list_of_mins_pos[-1]
    counter_max_list = list(range(len(list_of_maxs_pos)))
    # перебираем пары значений из списка экстремумов
    for i, j, separ in  zip(list_of_maxs[:-1], list_of_maxs_pos[:-1], counter_max_list): 
        k = (working_max - i)/(working_max_pos - j)
        b = working_max - k*working_max_pos
        list_kasanie_max = [] # временный список значений x, в которых паттерн контактирует
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
    # перебираем пары значений из списка экстремумов        
    for i, j, separ in  zip(list_of_mins[:-1], list_of_mins_pos[:-1], counter_max_list): # перебираем пары значений из списка экстремумов
        k = (working_min - i)/(working_min_pos - j)
        b = working_min - k*working_min_pos
        list_kasanie_min = [] # временный список значений x, в которых паттерн контактирует
        for ii, jj in  zip(list_of_mins[separ:-1], list_of_mins_pos[separ:-1]): 
            if (k * jj + b - ii) > zazor:
                list_kasanie_min = []
            elif  (ii - (k * jj + b)) > zazor:
                pass
            else:
                list_kasanie_min.append(jj)
        if len(list_kasanie_min) >= 4 and (list_kasanie_min[-1] - list_kasanie_min[0] > min_trend_len):
            list_kasanie_min.append(working_min_pos)
            memorized_set.add((2, list_kasanie_min[0], list_kasanie_min[-1], k ,b,len(list_kasanie_min)))
    return

historyfile = open(output_pure_file_name, 'a',newline='')
try:
    while start_walking <= pitstop:# обходим историю из прошлого в наст.
        zazor = zazo_coeff*price(start_walking)
        func1_result = _func(start_walking, zazor, historyfile)    
        start_walking += step_read
        print(start_walking)
except: 
    memorized_list = list(memorized_set) 
    memorized_list = [list(x) for x in memorized_list] 
    memorized_list.sort(key=lambda x: x[1])
#    вводим заголовок
    historyfile = open(output_pure_file_name, 'a', newline='')
    writer = csv.writer(historyfile, delimiter=',')
    writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
    writer.writerows(memorized_list)
    historyfile.close()    
memorized_list = list(memorized_set) 
memorized_list = [list(x) for x in memorized_list] 
memorized_list.sort(key=lambda x: x[1])
#######################вводим заголовок########################################
historyfile = open(output_pure_file_name,'a',newline='')
writer = csv.writer(historyfile, delimiter=',')
writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
writer.writerows(memorized_list)
historyfile.close()   