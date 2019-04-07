# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 22:25:27 2018

@author: user_PC
"""
import csv
#import operator
#import os
#import time
#import itertools
import math
#import ast
#from collections import Counter
import numpy as np
import pandas
inputpath='C:\\Users\\user_PC\\Desktop\\rts\\pureRTS18.csv'
historyOutPath="C:\\Users\\user_PC\\Desktop\\rts\\"
columns = ['<TIME>', '<VOLUME>', '<PRICE>']
###############################################################################
'''         header = 1  !!!    '''
data = pandas.read_csv(inputpath, sep = ',', names=columns, header = 0)

y_column = data['<PRICE>']
y_list = np.array(y_column)
x_list = np.array(range(len(y_list)))
ticks_total = len(y_list) - 1          
#########____I_____##########
step_read = 60
Lmax = 400000
zazo_coeff = 0.00005
relax_coef = 1
pitstop = ticks_total - 1000
###############################################################################
name_of_lines_file = "shit.csv"
###############################################################################
finish1 = 10560000 # 1000000 
###############################################################################
global x_stop
global direction
direction = 456
x_stop = 0
###############################################################################
'''##################НАЧАЛО ОПРЕДЕЛЕНИЯ FUNC1 ##################'''
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
    number_of_bars = 800
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
    for i, j, separ in  zip(list_of_maxs[:-1], list_of_maxs_pos[:-1], counter_max_list): # перебираем пары значений из списка максимумов
        k = (working_max - i)/(working_max_pos - j)
        b = working_max - k*working_max_pos
        list_kasanie_max = [] # временный список значений абсцыссы, в которых касательная касается графика в 3 и более точках
        for ii, jj in  zip(list_of_maxs[separ:-1], list_of_maxs_pos[separ:-1]): # перебираем пары значений из списка максимумов
            if (ii - (k * jj + b)) > zazor:
                list_kasanie_max = []
            elif (k * jj + b - ii) > zazor:
                pass
            else:
                list_kasanie_max.append(jj)
        if len(list_kasanie_max) >= 4 and (list_kasanie_max[-1] - list_kasanie_max[0] > 3000) :
            list_kasanie_max.append(working_max_pos)
            memorized_set.add((1, list_kasanie_max[0], list_kasanie_max[-1], k ,b, len(list_kasanie_max)))
#            csv.writer(historyfile).writerow([1, list_kasanie_max[0], list_kasanie_max[-1], k ,b, var, len(list_kasanie_max)])
    ############################################################################
    for i, j, separ in  zip(list_of_mins[:-1], list_of_mins_pos[:-1], counter_max_list): # перебираем пары значений из списка максимумов
        k = (working_min - i)/(working_min_pos - j)
        b = working_min - k*working_min_pos
        list_kasanie_min = [] # временный список значений абсцыссы, в которых касательная касается графика в 3 и более точках
        for ii, jj in  zip(list_of_mins[separ:-1], list_of_mins_pos[separ:-1]): # перебираем пары значений из списка максимумов
            if (k * jj + b - ii) > zazor:
                list_kasanie_min = []
            elif  (ii - (k * jj + b)) > zazor:
                pass
            else:
                list_kasanie_min.append(jj)
        if len(list_kasanie_min) >= 4 and (list_kasanie_min[-1] - list_kasanie_min[0] > 3000):
            list_kasanie_min.append(working_min_pos)
            memorized_set.add((2, list_kasanie_min[0], list_kasanie_min[-1], k ,b,len(list_kasanie_min)))
#            csv.writer(historyfile).writerow([2, list_kasanie_min[0], list_kasanie_min[-1], k ,b,len(list_kasanie_min)])
    return
###############################################################################
#                         ''' СОХРАНЯЛКА ЛИНИЙ '''
###############################################################################
breaker = finish1

################################################################################    
historyfile = open(historyOutPath + name_of_lines_file, 'a',newline='')
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
    historyfile = open(historyOutPath + name_of_lines_file,'a',newline='')
    writer = csv.writer(historyfile, delimiter=',')
    writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
    writer.writerows(memorized_list)
    historyfile.close()    
memorized_list = list (memorized_set) 
memorized_list = [list(x) for x in memorized_list] 
memorized_list.sort(key=lambda x: x[1])
#######################вводим заголовок########################################
historyfile = open(historyOutPath + name_of_lines_file,'a',newline='')
writer = csv.writer(historyfile, delimiter=',')
writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
writer.writerows(memorized_list)
historyfile.close()    