# -*- coding: utf-8 -*-
"""
двухпроцессорная система
на правом компе
евродоллар user_PC
работает

@author: user
"""
from multiprocessing import Process, Queue
import csv
import math
#import os
#import logging
import datetime
import time
import itertools
import numpy as np
import pandas
colnames = ['date and time','bid','ask','volume']

l__l = 175
p_angle = 0
n_angle = 0
Lmax = 80000
Lmin = 2700
Imin = 9500
relax_coef = 0.8
zazor = 1.71 # 2 для 27000для sbrf
wind = 68 # окно поиска экстремума на большом отрезке
wind_little = 8 # окно поиска экстремума на малом отрезке
peakgrad = 4.28 # 5 на 27000острота пиков на большом отрезке
peakgrad_little = 0 ## острота пиков на малом отрезке
proboy = 3.42 # во сколько раз должно отклонение цены от уровня превысить зазор, чтобы считать это пробоем (2 зазора обычно)
def func1(Lmin, Lmax, Imin, relax_coef, zazor, wind, wind_little, peakgrad, peakgrad_little, q):
    list_of_results_func1_send = []
    global previous_end_window
    previous_end_window = 300000
    while 1:
        try:
            data = pandas.read_csv('C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\Ticks.csv', names=colnames, header = 10)
            cols = data.columns
            y_column = data[cols[3]]
            y_list = list(y_column)
            number_of_strings = len(y_list)
            x_list = list(range(number_of_strings))
            def price(some): # ввел функцию цены акции от тика, индекс начинается с единицы
                return (y_list[some])
            global start_short_triangle
            end_short_triangle = number_of_strings
            if  end_short_triangle >= Lmin and ( end_short_triangle <= Lmax + 50):
                start_short_triangle = 50
            elif  end_short_triangle > (Lmax + 50):
                start_short_triangle =  end_short_triangle - Lmax
            y_list_whole_part = y_list[start_short_triangle: end_short_triangle]
            y_list_little_part = y_list[previous_end_window - 40: end_short_triangle -1]
            x_list_whole_part = x_list[start_short_triangle: end_short_triangle]
            x_list_little_part = x_list[previous_end_window - 40: end_short_triangle -1]
            
            _max, _min = peakdetect(y_list_whole_part,x_list_whole_part, wind, peakgrad)
            _max_little_part, _min_little_part = peakdetect(y_list_little_part,x_list_little_part, wind_little, peakgrad_little)
            previous_end_window = number_of_strings
            max_listx= [x[0] for x in _max] 
            min_listx= [x[0] for x in _min] 
            list_of_couples_max = list(itertools.product(_max, _max_little_part)) # выдает список сочетаний пар (xi,yi)
            list_of_couples_min = list(itertools.product(_min, _min_little_part))
            list_of_lists_max_3andmore_kasanie = [] #пустой список, потом он заполняется
            list_of_lists_min_3andmore_kasanie = []
            # сделали пары значений. выше этого момента ничего не меняем. пока  
#            def index_start_list (list_of_values, start_value): # функция, определяющая индекс элемента в простом списке по элементу
#                return list_of_values.index(start_value)
            for double_couple_max in  list_of_couples_max: # перебираем пары значений из списка максимумов
                list_double_couple_max = list(double_couple_max)
                np_double_couple_max = np.array(list_double_couple_max, dtype = np.dtype('int, float'))
                abscissa_list_max = np.array(np_double_couple_max["f0"], dtype = int) # список значений абсциссы abscissa_list_max[1] - старее,чем abscissa_list_max[0]
                ordinata_list_max = np.array(np_double_couple_max['f1'], dtype = float) # список значений ординаты
                A = np.vander(abscissa_list_max, 2)
                coeff_max, sse_max, rank_max , sing_a_max = np.linalg.lstsq(A,ordinata_list_max)
        #       coeff = [a,b] где coeff[0] - это тангенс угла наклона, а coeff[1] - +b 
                list_kasanie_max = [] # временный список значений абсцыссы, в которых касательная касается графика в 3 и более точках
                for i in max_listx: # сбор касаний я начинаю с первой точки
                   if (start_short_triangle < i <  end_short_triangle) and (i >= abscissa_list_max[0]) and (i <= abscissa_list_max[1]):
                       if (price(i) - (float(coeff_max[0] * i) + float(coeff_max[1]))) > zazor:
                           list_kasanie_max = []
                       elif (float(coeff_max[0] * i) + float(coeff_max[1]) - price(i)) > zazor:
                           pass
                       else:
                           list_kasanie_max.append(i)
                list_kasanie_max.append(list(coeff_max))
                if len(list_kasanie_max) >= 5 and (list_kasanie_max[-2] - list_kasanie_max[0]) > (Lmin):
                    list_of_lists_max_3andmore_kasanie.append(list_kasanie_max)
        ##################################################################################################          
            for double_couple_min in  list_of_couples_min:
                list_double_couple_min = list(double_couple_min)
                np_double_couple_min = np.array(list_double_couple_min, dtype = np.dtype('int, float'))
                abscissa_list_min = np.array(np_double_couple_min["f0"], dtype = int) # список значений абсциссы
                ordinata_list_min = np.array(np_double_couple_min['f1'], dtype = float) # список значений ординаты
                A = np.vander(abscissa_list_min, 2)
                coeff_min, sse_min, rank_min , sing_a_min = np.linalg.lstsq(A,ordinata_list_min)
        #       coeff = [a,b] где coeff[0] - это тангенс угла наклона, а coeff[1] - +b 
                list_kasanie_min = [] # временный список значений абсцbссы, в которых касательная касается графика в 3 и более точках
                for i in min_listx: # сбор касаний я начинаю с первой точки
                    if (start_short_triangle < i <  end_short_triangle) and (i >= abscissa_list_min[0]) and (i <= abscissa_list_min[1]):
                        if (price(i) - (float(coeff_min[0] * i) + float(coeff_min[1]))) > zazor:
                            pass
                        elif (float(coeff_min[0] * i) + float(coeff_min[1]) - price(i)) > zazor:
                            list_kasanie_min = []
                        else:
                            list_kasanie_min.append(i) # собираю временный список абсцисс касаний 
                list_kasanie_min.append(list(coeff_min))
                if len(list_kasanie_min) >= 5 and (list_kasanie_min[-2] - list_kasanie_min[0]) > (Lmin):
                    list_of_lists_min_3andmore_kasanie.append(list_kasanie_min)
        ##########################################################################################################                   
            def max_angle(inputlist): # выбор максимального тангенса угла для нижней прямой
                return max([sublist[-1] for sublist in inputlist])#
            def min_angle(inputlist): # выбор минимального тангенса угла для верхней прямой
                return min([sublist[-1] for sublist in inputlist])#
            def max_element(inputlist): # выбор  по наличию точки в линии с максимальной абсциссой
                return max([sublist[-1] for sublist in inputlist])#
            def length_of_line(somelist):
                return (somelist[-1] - somelist[0])
            def list_of_just_dot_in_general_list(inputlist): # оставляет только список точек
                return ([sublist[:-1] for sublist in inputlist])
            def length (dot1, dot2): # функция вычисления расстояний между точками
                return (np.linalg.norm(dot1-dot2))
            def importance_of_line (number_of_dots,  length, dispersion_of_distance): # коэффициент значимости линии 
                return(number_of_dots**2 *  length/(dispersion_of_distance**5))
        ###############################################################################
            min_list_of_duplicates = list_of_just_dot_in_general_list(list_of_lists_min_3andmore_kasanie)
            min_set = set(tuple(x) for x in min_list_of_duplicates)
            min_list_abscissa = [list(x) for x in min_set] # это список абсцисс
            min_list_ordinata = [] # индексы совпадают
            min_list_coordinates = [] # индексы совпадают
            list_of_var_min = [] # индексы совпадают
            importance_min = [] # индексы совпадают
            list_angle_min = [] # индексы совпадают
            for i in min_list_abscissa:
                c = []
                d = []
                for ii in i:
                    c.append(price(ii))
                    d.append(price(ii))
                min_list_coordinates.append(c)
                min_list_ordinata.append(d)
            real_distance_min_list = []    
            for i, ii in zip (min_list_abscissa, min_list_ordinata):
                real_distance_min_list.append(math.hypot(i[-1] - i[0], ii[-1] - ii[0]))
                A = np.vander(i, 2)
                coeff_min, sse_min, rank_min , sing_a_min = np.linalg.lstsq(A, ii)
                list_angle_min.append(coeff_min)
            list_of_var_min = []
            for i in min_list_abscissa:
                c =  [x-x2 for x, x2 in zip(i[1:], i[:-1])]
                list_of_var_min.append(max(c)/np.mean(c))
            for i1, i2, i5 in zip(min_list_abscissa, list_of_var_min, real_distance_min_list):
                importance_min.append(importance_of_line((len(i1)), i5, i2))
        ####################################################################################################################
            max_list_of_duplicates = list_of_just_dot_in_general_list(list_of_lists_max_3andmore_kasanie)
            max_set = set(tuple(x) for x in max_list_of_duplicates)
            max_list_abscissa = [ list(x) for x in max_set ] # это список абсцисс
            max_list_ordinata = [] # индексы совпадают
            max_list_coordinates = [] # индексы совпадают
            list_of_var_max = [] # индексы совпадают
            importance_max = [] # индексы совпадают
            list_angle_max = [] # индексы совпадают
            for i in max_list_abscissa:
                c = []
                d = []
                for ii in i:
                    c.append(price(ii))
                    d.append(price(ii))
                max_list_coordinates.append(c)
                max_list_ordinata.append(d)
            real_distance_min_list = []  
            for i, ii in zip (max_list_abscissa, max_list_ordinata):
                A = np.vander(i, 2)
                coeff_max, sse_max, rank_max , sing_a_max = np.linalg.lstsq(A, ii)
                real_distance_min_list.append(math.hypot(i[-1] - i[0], ii[-1] - ii[0]))
                list_angle_max.append(coeff_max)
            list_of_var_max = [] 
            for i in max_list_abscissa:
                c =  [x-x2 for x, x2 in zip(i[1:], i[:-1])]
                list_of_var_max.append(max(c)/np.mean(c))
            for i1, i2, i5 in zip(max_list_abscissa, list_of_var_max, real_distance_min_list):
                importance_max.append(importance_of_line((len(i1)), i5, i2))
        #####################################################################################################  
            final_list_lower = []
            final_list_upper = []
            a = 0
            b = 0
            la = 0
            lb = 0
            fa = 0
            fb = 0
            the_best_max_index_value = 0
            the_best_min_index_value = 0
            if len(importance_min) > 0:
                b = max(importance_min)
                if b > Imin:
                    the_best_min = b
                    the_best_min_index = ([i for i, j in enumerate(importance_min) if j == the_best_min])
                    the_best_min_index_value = the_best_min_index[0]
                    fb = min_list_abscissa[the_best_min_index_value][0]
                    lb = min_list_abscissa[the_best_min_index_value][-1]
                else:
                    b = 0
            if len(importance_max) > 0:
                a = max(importance_max)
                if a > Imin:
                    the_best_max = a
                    the_best_max_index = ([i for i, j in enumerate(importance_max) if j == the_best_max]) # лучший % по важности (дает хор результат)
                    the_best_max_index_value = the_best_max_index[0]
                    fa = max_list_abscissa[the_best_max_index_value][0]
                    la = max_list_abscissa[the_best_max_index_value][-1]
                else:
                    a = 0
        ###########################################################################################################
            if a > 0 or b > 0:
                if a > b:
                    if list_angle_max[the_best_max_index[0]][0] < p_angle:
                        length_max = la - fa
                        final_list_upper = [fa, la, length_max, the_best_max, list_angle_max[the_best_max_index[0]], 1]
                        if final_list_upper[:2] not in list_of_results_func1_send:
#                            list_of_results_func1.append(final_list_upper)
                            list_of_results_func1_send.append(final_list_upper[:2])
#                            list_of_results_func1_without_dublers = dict((x[0], x) for x in list_of_results_func1).values()
#                            list_of_results_func1_without_dublers = list(list_of_results_func1_without_dublers)
                            q.put(final_list_upper)
                            time.sleep(0.0001)
                if a < b:
                    if list_angle_min[the_best_min_index[0]][0] > n_angle:
                        length_min = lb - fb
                        final_list_lower = [fb, lb, length_min, the_best_min, list_angle_min[the_best_min_index[0]], 2]
                        if final_list_lower[:2] not in list_of_results_func1_send:
#                            list_of_results_func1.append(final_list_lower)
                            list_of_results_func1_send.append(final_list_lower[:2])
    #                        list_of_results_func1_without_dublers = dict((x[0], x) for x in list_of_results_func1).values()
    #                        list_of_results_func1_without_dublers = list(list_of_results_func1_without_dublers)
                            q.put(final_list_lower)
                            time.sleep(0.0001)
        except:
            ValueError    
######################################################################################################################
def peakdetect(y_axis, x_axis, lookahead, delta):
    maxtab = []
    mintab = []
    dump = []   #Used to pop the first hit which always if false
    length = len(y_axis)
    if x_axis is None:
        x_axis = range(length)
    #needs to be a numpy array
    y_axis = np.asarray(y_axis)
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                maxtab.append((mxpos, mx))
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                mintab.append((mnpos, mn))
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            maxtab.pop(0)
            #print "pop max"
        else:
            mintab.pop(0)
            #print "pop min"
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
    
    return maxtab, mintab                                   
######################################################################################################################
###################################----  П Р О В Е Р К А    Н А    П Р О Б О Й ---- ##################################
######################################################################################################################
def func2(q):
    list_of_results = []
    list_of_abscissa_deals = [0]
    while 1:
        if not q.empty():
            gained_list = []
            gained_list = q.get()
            if gained_list not in list_of_results:
                list_of_results.append(gained_list)
            data = pandas.read_csv('C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\Ticks.csv', names=colnames, header = 10)
            cols = data.columns
            y_column = data[cols[3]]
            y_list = list(y_column)
            last_max_price = float(max(y_list[-5:]))
            last_min_price = float(min(y_list[-5:]))
            lenght_of_data = len(y_list)
            for best_result in list_of_results:
                last_abscissa_deal = list_of_abscissa_deals[-1]
                marker = best_result[-1]
                final_list = best_result
                L = best_result[2]
                end = final_list[1]  
                if (end + relax_coef*L) >= lenght_of_data and (lenght_of_data - last_abscissa_deal) > l__l:
                    if marker == 1:
                        if last_max_price > ((final_list[4][0])*lenght_of_data + final_list[4][1] + proboy) and final_list[4][0] <= 0: # пробой больше спреда и ведущий тренд бычий, а сопротивление не слишком бычая
                            dofile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\dofile.csv", 'a',newline='')
                            csv.writer(dofile).writerow([1, last_max_price])
                            dofile.close()
                            historyfile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\history.csv", 'a',newline='')
                            csv.writer(historyfile).writerow([datetime.datetime.now(), lenght_of_data , 1, final_list, last_max_price])
                            historyfile.close()
                            list_of_results.remove(best_result)
                            list_of_abscissa_deals.append(lenght_of_data)
                        else:
                            pass
                    if marker == 2:
                        if last_min_price < ((final_list[4][0])*lenght_of_data + final_list[4][1] - proboy) and final_list[4][0] >= 0: # пробой больше спреда и ведущий тренд бычий, а сопротивление не слишком бычая
                            dofile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\dofile.csv", 'a',newline='')
                            csv.writer(dofile).writerow([2, last_min_price])
                            dofile.close()
                            historyfile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\history.csv", 'a',newline='')
                            csv.writer(historyfile).writerow([datetime.datetime.now(), lenght_of_data , 2, final_list, last_min_price])
                            historyfile.close()
                            list_of_results.remove(best_result)
                            list_of_abscissa_deals.append(lenght_of_data)
                        else:
                            pass
                else:
                    list_of_results.remove(best_result)
        else:
            data = pandas.read_csv('C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\Ticks.csv', names=colnames, header = 10)
            cols = data.columns
            y_column = data[cols[3]]
            y_list = list(y_column)
            last_max_price = float(max(y_list[-5:]))
            last_min_price = float(min(y_list[-5:]))
            lenght_of_data = len(y_list)
            for best_result in list_of_results:
                last_abscissa_deal = list_of_abscissa_deals[-1]
                marker = best_result[-1]
                final_list = best_result
                L = best_result[2]
                end = final_list[1]  
                if (end + relax_coef*L) >= lenght_of_data and (lenght_of_data - last_abscissa_deal) > l__l:
                    if marker == 1:
                        if last_max_price > ((final_list[4][0])*lenght_of_data + final_list[4][1] + proboy) and final_list[4][0] <= 0: # пробой больше спреда и ведущий тренд бычий, а сопротивление не слишком бычая
                            dofile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\dofile.csv", 'a',newline='')
                            csv.writer(dofile).writerow([1, last_max_price])
                            dofile.close()
                            historyfile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\history.csv", 'a',newline='')
                            csv.writer(historyfile).writerow([datetime.datetime.now(), lenght_of_data , 1, final_list, last_max_price])
                            historyfile.close()
                            list_of_results.remove(best_result)
                            list_of_abscissa_deals.append(lenght_of_data)
                        else:
                            pass
                    if marker == 2:
                        if last_min_price < ((final_list[4][0])*lenght_of_data + final_list[4][1] - proboy) and final_list[4][0] >= 0: # пробой больше спреда и ведущий тренд бычий, а сопротивление не слишком бычая
                            dofile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\dofile.csv", 'a',newline='')
                            csv.writer(dofile).writerow([2, last_min_price])
                            dofile.close()
                            historyfile = open("C:\\Users\\user\\AppData\\Roaming\\MetaQuotes\\Terminal\\62C110D4502B034046D10450DFB69011\\MQL5\\Files\\history.csv", 'a',newline='')
                            csv.writer(historyfile).writerow([datetime.datetime.now(), lenght_of_data , 2, final_list, last_min_price])
                            historyfile.close()
                            list_of_results.remove(best_result)
                            list_of_abscissa_deals.append(lenght_of_data)
                        else:
                            pass
                else:
                    list_of_results.remove(best_result)
#######################################################################################################################################
###########################################################################################################################
if __name__ == '__main__':
    global q# это вроде не обязательно
    q = Queue()
    process_one = Process(target=func1, args=(Lmin, Lmax, Imin, relax_coef, zazor, wind, wind_little, peakgrad, peakgrad_little, q))
    process_two = Process(target=func2, args=(q,))
    process_one.start()
    process_two.start()
    q.close()
    q.join_thread()
    process_one.join()
    process_two.join()
            
            