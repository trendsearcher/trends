# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 13:16:56 2018

@author: user_PC
"""
# -*- coding: utf-8 -*-
"""
"""
import csv
import matplotlib.pyplot as plt
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas
inputpath='C:\\Users\\user_PC\\Desktop\\rurusd2\\USDRUB_TOM_2018_07_10pure.csv'
historyOutPath="C:\\Users\\user_PC\\Desktop\\rurusd2\\"
###############################################################################
normal_trends = "normal_trends_half_zazor_crossed_"
broken_trends = "broken_trends_half_zazor_crossed_"
################################################################################
trendspath="C:\\Users\\user_PC\\Desktop\\rurusd2\\normal_trends_half_zazor2.csv"
colnames = ['<price_eba>']
###############################################################################
###############################################################################
def peakdetect(y_axis, x_axis, lookahead, delta):
    maxtab = []
    mintab = []
    dump = []   #Used to pop the first hit which always if false
    length = len(y_axis)
    if x_axis is None:
        x_axis = range(length)
    y_axis = np.asarray(y_axis)
    mn, mx = np.Inf, -np.Inf
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        if y < mx-delta and mx != np.Inf:
            if y_axis[index:index+lookahead].max() < mx:
                maxtab.append((mxpos, mx))
                dump.append(True)
                mx = np.Inf
                mn = np.Inf
        if y > mn+delta and mn != -np.Inf:
            if y_axis[index:index+lookahead].min() > mn:
                mintab.append((mnpos, mn))
                dump.append(False)
                mn = -np.Inf
                mx = -np.Inf
    try:
        if dump[0]:
            maxtab.pop(0)
        else:
            mintab.pop(0)
        del dump
    except IndexError:
        pass
    return maxtab, mintab                
##########################################
def slisingoldtail(commonlist, boarder):
    if len(commonlist) >= 2:
        if commonlist[0] < boarder:
            while commonlist[0] < boarder:
                del commonlist[0]
            return(commonlist)   
        else:    
            return(commonlist)   
    else:        
        return(commonlist)    
###############################################################################
def price(some): # ввел функцию цены акции от тика, индекс начинается с единицы
    return (y_list[some])
def length (dot1, dot2): # функция вычисления расстояний между точками
    return (np.linalg.norm(dot1-dot2))
def importance_of_line (number_of_dots,  length, dispersion_of_distance): # коэффициент значимости линии 
    return(number_of_dots *  length/(dispersion_of_distance))
def find_zazor(pr, st):# зазор зависит от текущей цены и длины тренда (длину усредняем для каждой стадии)
    if st == 2:
#        return (0.00005 + 0.0000732*pr)
        return (0 + 0.0000366*pr)
    if st == 3:
#        return (0.00005 + 0.000876*pr)
        return (0 + 0.000438*pr)
    if st == 4:
#        return(0.00005 + 0.00272*pr)
        return (0 + 0.00136*pr)
###############################################################################
'''         header = 1  !!!    '''
data = pandas.read_csv(inputpath, names=colnames, sep = '\t', header = 1)
cols = data.columns
y_column = data[cols[0]]
y_list = list(y_column)
x_list = list(range(len(y_list)))
''' stage = 2, 3, 4 '''
'''*******************'''
stage = 2
'''*******************'''
output_broken = broken_trends + str(stage) + '.csv'
output_normal = normal_trends + str(stage) + '.csv'
relax_coeff = 1# 100% длины в будущее
ticks_total = len(y_list) - 1          
relax_coef = 1
Imin = 0
deep_of_trend_brokiness = 3 # соотношение количества найденных экстремумов под линиией и внутри
###############################################################################
if stage ==2:
    Lmin = 590
    wind = 35 # окно поиска экстремума на большом отрезке
######____III_____#########
if stage == 3:
    Lmin = 35800
    wind = 420 # окно поиска экстремума на большом отрезке
######____IV_____#########
if stage == 4:
    Lmin = 401440
    wind = 1296 # окно поиска экстремума на большом отрезке
###############################################################################
##########################################################################
counter_list = []
with open (trendspath , "r") as csvfile:
    reader = csv.reader(csvfile, delimiter =  ",")
    my_list0 = list(reader)
    my_list_chisto_chisto = []
    for i in my_list0:
        direction_of_deal = int(i[0])
        basic_parameters = i[1].split('[[')[1].split(',')
        first_dot = int(basic_parameters[0])
        last_dot = int(basic_parameters[1])
        leng = last_dot - first_dot
        I = float(basic_parameters[2])
        angle = float(basic_parameters[3].split('([')[1])
        b_coeff = float(basic_parameters[4].split('])')[0])
        if I > 5 and leng > 600: # отбрасываем всякую шнягу
            if (direction_of_deal == 1 and angle <= 0) or (direction_of_deal == 2 and angle >=0):
                counter_list.append([first_dot, last_dot, angle, b_coeff, direction_of_deal])
for i in counter_list:
    direction = i[-1]
    start_dot_2 = i[0]
    end_dot_2 = i[1]
    angle_2 = i[2]
    b_coeff_2 = i[3]
    relaxed_end_dot = relax_coeff*(end_dot_2 - start_dot_2) + end_dot_2
    zazor = find_zazor(price(end_dot_2), stage)
    peakgrad = 5*zazor
    proboy = 3*zazor
    last_touching_list = []
    breakout_list = []
    for ii in x_list[end_dot_2: relaxed_end_dot]:
        if direction == 1 :
            if abs(price(ii) - (ii*angle_2 + b_coeff_2)) <= zazor:
                last_touching_list.append(ii)
            if price(ii) - (ii*angle_2 + b_coeff_2) > proboy:
                breakout_list.append(ii)
                break
        if direction == 2 :
            if abs(price(ii) - (ii*angle_2 + b_coeff_2)) <= zazor:
                last_touching_list.append(ii)
            if (ii*angle_2 + b_coeff_2) - price(ii) > proboy:
                breakout_list.append(ii)
                break
    if len(last_touching_list) > 0 and len(breakout_list) != 0:# если было касание перед пробоем и был пробой
        breakout = breakout_list[0] 
        end_dot_3 =  last_touching_list[-1]
        print(end_dot_3)
########################################################################
        y_list_whole_part = y_list[start_dot_2 : end_dot_3]
        x_list_whole_part = x_list[start_dot_2: end_dot_3]
        _max, _min = peakdetect(y_list_whole_part,x_list_whole_part, wind, peakgrad)
        max_listx= [x[0] for x in _max] 
        min_listx= [x[0] for x in _min]
        list_of_lists_max_3andmore_kasanie = [] #список абсцисс непробитых трендов
        list_of_lists_max_3andmore_kasanie_broken = []#список абсцисс пробитых трендов
        list_of_lists_min_3andmore_kasanie = []
        list_of_lists_min_3andmore_kasanie_broken = []
################################################################################
        if direction == 1:
            list_double_couple_max = [(start_dot_2, price(start_dot_2)), (end_dot_2, price(end_dot_2))]
            np_double_couple_max = np.array(list_double_couple_max, dtype = np.dtype('int, float'))
            abscissa_list_max = np.array(np_double_couple_max["f0"], dtype = int) # список значений абсциссы abscissa_list_max[1] - старее,чем abscissa_list_max[0]
            ordinata_list_max = np.array(np_double_couple_max['f1'], dtype = float) # список значений ординаты
            A = np.vander(abscissa_list_max, 2)
            coeff_max, sse_max, rank_max , sing_a_max = np.linalg.lstsq(A,ordinata_list_max, rcond=-1)
#            coeff_max = [angle_2, b_coeff_2]
        #       coeff = [a,b] где coeff[0] - это тангенс угла наклона, а coeff[1] - +b 
            list_kasanie_max = [] # временный список значений абсцыссы, в которых касательная касается графика в 3 и более точках
            list_kasanie_max_broken = []
            dots_above_line_max = 0 # вышли за пределы линии
            dots_under_line_max = 0
            for i in max_listx: # сбор касаний я начинаю с первой точки
               if (price(i) - (float(coeff_max[0] * i) + float(coeff_max[1]))) > zazor:
                   dots_above_line_max += 1
                   list_kasanie_max = []
               elif (float(coeff_max[0] * i) + float(coeff_max[1]) - price(i)) > zazor:
                   dots_under_line_max += 1
               else:
                   list_kasanie_max.append(i)
                   list_kasanie_max_broken.append(i)
            list_kasanie_max.append(end_dot_3)       
            list_kasanie_max_broken.append(end_dot_3)       
            if  (list_kasanie_max[-1] - list_kasanie_max[0]) > Lmin and len(list_kasanie_max) > 4: #тренд захватывает последнюю точку и длиннее минимума
                list_of_lists_max_3andmore_kasanie.append(list_kasanie_max)
                if list_kasanie_max_broken != list_kasanie_max and (dots_under_line_max/dots_above_line_max) > deep_of_trend_brokiness:
                    list_of_lists_max_3andmore_kasanie_broken.append(list_kasanie_max_broken)  
        #__несломанные линии поддержки___#############################################
        max_set = set(tuple(x) for x in list_of_lists_max_3andmore_kasanie)
        max_list_abscissa = [list(x) for x in max_set] # это список абсцисс # индексы совпадают
        #   _max  пары (x,y) max_listx (x)
        max_list_ordinata = [] # индексы совпадают
        list_of_var_max = [] # индексы совпадают
        importance_max = [] # индексы совпадают
        importance_max_val = [] # индексы совпадают
        list_angle_max = [] # индексы совпадают
        list_sse_max = []# индексы совпадают
        real_distance_max_list = [] # индексы совпадают  
        ##__списки со списками параметров вершин и пиков, таких как высоты, основания, острота и т.д.
        tops_max_parameters = [] # индексы совпадают/ список списков [list_peaks, list_of_peaks_H, list_of_peak_W]
        peak_max_parameters = [] # индексы совпадают/ список списков [list_tops, list_of_tops_H, list_of_tops_W]
        ####____определяем уравнение тренда____####################################
        for i in max_list_abscissa: # i - список абсцисс касаний вида [x1, x2, x3..]
            d = []
            for ii in i:
                d.append(price(ii))
            max_list_ordinata.append(d)
        for i, ii in zip (max_list_abscissa, max_list_ordinata):
            list_tops = []      # список вида [(x1, y1), (x2, y2)---] с координатами вершины
            list_of_tops_H = []    # список вида [x1, x2, ...] с высотой вершины отн линии
            list_of_tops_W = [] # список вида [x1, x2, ...] с шириной основания вершины
            real_distance_max_list.append(math.hypot(i[-1] - i[0], ii[-1] - ii[0]))
            A = np.vander(i, 2)
            coeff_max, sse_max, rank_max , sing_a_max = np.linalg.lstsq(A, ii, rcond=-1)
#            coeff_max = [angle_2, b_coeff_2]
            list_angle_max.append(coeff_max)
            list_sse_max.append(sse_max)
            amount_of_peaks_on_line = len(i)
            for l in range(amount_of_peaks_on_line - 1):
                k = i[l +1]
                j = i[l]
                min_selected = [x for x in _min if (x[0] >= j and x[0] <= k)]
                if len(min_selected) > 0:# найдена нормальная вершина в этом интервале
                    min_selected_y = [x[1] for x in min_selected]
                    local_top_y = min(min_selected_y)
                    list_of_tops_W.append((k-j))
                    local_top = [x for x in min_selected if x[1] == local_top_y][0]
                    list_tops.append(local_top)
                    local_top_H =  coeff_max[0]*local_top[0] + coeff_max[1] - local_top_y
                    list_of_tops_H.append(local_top_H)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        list_of_tops_H.insert(0, local_top_H)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        list_of_tops_H.append(local_top_H)
                else: # вершина в интервале не найдена. будем использовать середину интервала
                    min_selected = int((k + j)/2)
                    local_top_y = price(min_selected)
                    list_of_tops_W.append((k-j))
                    local_top = (min_selected, local_top_y)
                    list_tops.append(local_top)
                    local_top_H =   coeff_max[0]*local_top[0] + coeff_max[1] - local_top_y
                    list_of_tops_H.append(local_top_H)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        list_of_tops_H.insert(0, local_top_H)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        list_of_tops_H.append(local_top_H)
            tops_sharpness = [(x/y) for x,y in zip(list_of_tops_H, list_of_tops_W)] 
            tops_max_parameters.append([list_tops, list_of_tops_H, list_of_tops_W, tops_sharpness])
        for i, j, k in zip(max_list_abscissa, tops_max_parameters, max_list_ordinata): #между образующими тренд пиками есть по 1 вершине
            list_peaks = [(x, y) for x,y in zip(i, k)]      # список вида [(x1, y1), (x2, y2)---] с координатами пиков
            list_of_peak_H = [] # список вида [x1, x2, ...] с высотой пиков отн линии соседних вершин
            list_of_peak_W = [] # список вида [x1, x2, ...] с шириной пиков по соседним вершинам  
            list_peaks = [(x, y) for x,y in zip(i, k)]
            jj = j[0]
            for l in range(len(i)):
                peak_x = i[l]
                peak_y = k[l]
                first_neighbour_top = jj[l]
                second_neighbour_top = jj[l + 1]
                peak_W = second_neighbour_top[0] - first_neighbour_top[0]
                tg_angle = (second_neighbour_top[1] - first_neighbour_top[1])/(peak_W)
                b_coeff = first_neighbour_top[1] - tg_angle*first_neighbour_top[0]
                peak_H = peak_y - tg_angle*peak_x - b_coeff
                list_of_peak_W.append(peak_W)
                list_of_peak_H.append(peak_H)
            peak_sharpness = [(x/y) for x,y in zip(list_of_peak_H, list_of_peak_W)] 
            peak_max_parameters.append([list_peaks, list_of_peak_H, list_of_peak_W, peak_sharpness])
        for i in max_list_abscissa:
            c =  [x-x2 for x, x2 in zip(i[1:], i[:-1])]
            mean_distance = np.mean(c)
            cc = [(abs(x - mean_distance)) for x in c]
            list_of_var_max.append(np.mean(cc))
        for i1, i2, i5, i6 in zip(max_list_abscissa, list_of_var_max, real_distance_max_list, list_sse_max):
            importance_max.append(importance_of_line((len(i1)), i5, i2))
            importance_max_val.append([i1, i2, i5, i6])
    ##__сломанные линии поддержки___###############################################
        max_set_broken = set(tuple(x) for x in list_of_lists_max_3andmore_kasanie_broken)
        max_list_abscissa_broken = [list(x) for x in max_set_broken] # это список абсцисс # индексы совпадают
        max_list_ordinata_broken = [] # индексы совпадают
        list_of_var_max_broken = [] # индексы совпадают
        importance_max_broken = [] # индексы совпадают
        importance_max_val_broken = [] # индексы совпадают
        list_angle_max_broken = [] # индексы совпадают
        list_sse_max_broken = []# индексы совпадают
        ##__списки со списками параметров вершин и пиков, таких как высоты, основания, острота и т.д.
        tops_max_parameters_broken = [] # индексы совпадают/ список списков [list_peaks, list_of_peaks_H, list_of_peak_W]
        peak_max_parameters_broken = []
        integrals_of_broken_parts = [] # индексы совпадают/ список списков интегралов вышедших частей
        broken_parts_H = [] # индексы совпадают/ список списков высоты вышедших частей
        persentage_of_brokeness_max = [] # индексы совпадают/ отношение площади всех пробоев к площади вершин
        for i in max_list_abscissa_broken:
            d = []
            for ii in i:
                d.append(price(ii))
            max_list_ordinata_broken.append(d)
        real_distance_max_list_broken = []    
        for i, ii in zip (max_list_abscissa_broken, max_list_ordinata_broken):
            list_tops = []      # список вида [(x1, y1), (x2, y2)---] с координатами вершины
            list_of_tops_H = []    # список вида [x1, x2, ...] с высотой вершины отн линии
            list_of_tops_W = [] # список вида [x1, x2, ...] с шириной основания вершины
            real_distance_max_list_broken.append(math.hypot(i[-1] - i[0], ii[-1] - ii[0]))
            A = np.vander(i, 2)
            coeff_max, sse_max, rank_max , sing_a_max = np.linalg.lstsq(A, ii, rcond=-1)
#            coeff_max = [angle_2, b_coeff_2]
            list_angle_max_broken.append(coeff_max)
            list_sse_max_broken.append(sse_max)
            amount_of_peaks_on_line = len(i)
            for l in range(amount_of_peaks_on_line - 1):
                k = i[l +1]
                j = i[l]
                min_selected = [x for x in _min if (x[0] >= j and x[0] <= k)]
                if len(min_selected) > 0:# найдена нормальная вершина в этом интервале
                    min_selected_y = [x[1] for x in min_selected]
                    local_top_y = min(min_selected_y)
                    list_of_tops_W.append((k-j))
                    local_top = [x for x in min_selected if x[1] == local_top_y][0]
                    list_tops.append(local_top)
                    local_top_H = coeff_max[0]*local_top[0] + coeff_max[1] - local_top_y
                    if local_top_H >= 0:
                        list_of_tops_H.append(local_top_H)
                    else:
                        list_of_tops_H.append(0)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.insert(0, local_top_H)
                        else:
                            list_of_tops_H.insert(0, 0)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.append(local_top_H)
                        else:
                            list_of_tops_H.append(0)
                else: # вершина в интервале не найдена. будем использовать середину интервала
                    min_selected = int((k + j)/2)
                    local_top_y = price(min_selected)
                    list_of_tops_W.append((k-j))
                    local_top = (min_selected, local_top_y)
                    list_tops.append(local_top)
                    local_top_H = coeff_max[0]*local_top[0] + coeff_max[1] - local_top_y
                    if local_top_H >= 0:
                        list_of_tops_H.append(local_top_H)
                    else:
                        list_of_tops_H.append(0)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.insert(0, local_top_H)
                        else:    
                            list_of_tops_H.insert(0, 0)
                        list_of_tops_H.insert(0, local_top_H)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.append(local_top_H)
                        else:    
                            list_of_tops_H.append(0)
            tops_sharpness = [(x/y) for x,y in zip(list_of_tops_H, list_of_tops_W)] 
            tops_integral_rough = [(x*y/2) for x,y in zip(list_of_tops_H, list_of_tops_W)]
            tops_max_parameters_broken.append([list_tops, list_of_tops_H, list_of_tops_W, tops_sharpness, tops_integral_rough])
        for i, j, k in zip(max_list_abscissa_broken, tops_max_parameters_broken, max_list_ordinata_broken): #между образующими тренд пиками есть по 1 вершине
            list_peaks = [(x, y) for x,y in zip(i, k)]      # список вида [(x1, y1), (x2, y2)---] с координатами пиков
            list_of_peak_H = [] # список вида [x1, x2, ...] с высотой пиков отн линии соседних вершин
            list_of_peak_W = [] # список вида [x1, x2, ...] с шириной пиков по соседним вершинам  
            list_peaks = [(x, y) for x,y in zip(i, k)]
            jj = j[0]
            for l in range(len(i)):
                peak_x = i[l]
                peak_y = k[l]
                first_neighbour_top = jj[l]
                second_neighbour_top = jj[l + 1]
                peak_W = second_neighbour_top[0] - first_neighbour_top[0]
                tg_angle = (second_neighbour_top[1] - first_neighbour_top[1])/(peak_W)
                b_coeff = first_neighbour_top[1] - tg_angle*first_neighbour_top[0]
                peak_H = tg_angle*peak_x + b_coeff - peak_y
                list_of_peak_W.append(peak_W)
                list_of_peak_H.append(peak_H)
            peak_sharpness = [(x/y) for x,y in zip(list_of_peak_H, list_of_peak_W)] 
            peak_max_parameters_broken.append([list_peaks, list_of_peak_H, list_of_peak_W, peak_sharpness])
        for i in max_list_abscissa_broken:
            c =  [x-x2 for x, x2 in zip(i[1:], i[:-1])]
            mean_distance = np.mean(c)
            с_broken = [(abs(x - mean_distance)) for x in c]
            list_of_var_max_broken.append(np.mean(с_broken))
            list_of_proboys_max = [] # список значений x, где график пересекает тренд
            list_of_proboys_max_intervals = []
            first_dot = i[0]
            last_dot = i[-1] + 1
            if ((float(coeff_max[0] * last_dot) + float(coeff_max[1])) - zazor) < price(last_dot): # только если последняя точка в границах тренда:
                if ((float(coeff_max[0] * first_dot) + float(coeff_max[1])) - zazor) < price(first_dot):
                    marker_0 = 0 # первая точка в границах тренда 
                    marker_00 = 0
                else:
                    marker_0 = 1 # первая точка уехала за границы тренда
                    marker_00 = 1
                for ii in x_list[first_dot + 1:last_dot]:
                    marker_1 = marker_0
                    if ((float(coeff_max[0] * ii) + float(coeff_max[1])) - zazor) < price(ii):
                        marker_0 = 0
                    else:
                        marker_0 = 1
                    if marker_0 - marker_1 == 1: # 
                        list_of_proboys_max.append(ii)
                    if marker_0 - marker_1 == -1: # 
                        list_of_proboys_max.append(ii)
                list_of_proboys_max_val = len(list_of_proboys_max)        
                if marker_00 == 0 and list_of_proboys_max_val % 2 == 0: 
                   for i,k in zip(list_of_proboys_max[0::2], list_of_proboys_max[1::2]):
                       list_of_proboys_max_intervals.append([i,k])
                if marker_00 == 1 and list_of_proboys_max_val % 2 != 0: 
                   for i,k in zip(list_of_proboys_max[1::2], list_of_proboys_max[2::2]):
                       list_of_proboys_max_intervals.append([i,k])
            list_of_H_max = [] # высоты пробоев
            list_of_S_max = [] # площади пробоев
            list_of_SH_list_max = []
            list_of_W_max = []          
            for i in list_of_proboys_max_intervals:
                start = i[0]
                end = i[1] + 1
                local_w = (end - start -1)
                list_of_W_max.append(local_w)
                local_max_list = []
                for ii in x_list[start: end]:
                    local_max_list.append((float(coeff_max[0] * ii) + float(coeff_max[1])) - zazor - price(ii))
                list_of_SH_list_max.append(local_max_list)
            list_of_S_max = [sum(x) for x in list_of_SH_list_max] 
            list_of_H_max = [max(x) for x in list_of_SH_list_max]
            integrals_of_broken_parts.append(list_of_S_max)
            broken_parts_H.append(list_of_H_max)
        for i, j in zip(tops_max_parameters_broken, integrals_of_broken_parts):
            ii = i[-1]
            s2 = sum(ii)
            s1 = sum(j)
            s = s1/s2
            persentage_of_brokeness_max.append(s)
        for i1, i2, i5, i6 in zip(max_list_abscissa_broken, list_of_var_max_broken, real_distance_max_list_broken, list_sse_max):
            importance_max_broken.append(importance_of_line((len(i1)), i5, i2))
            importance_max_val_broken.append([i1, i2, i5, i6])    
    ##################################################################################################          
        if direction == 2:
            list_double_couple_min = [(start_dot_2, price(start_dot_2)), (end_dot_2, price(end_dot_2))]
            np_double_couple_min = np.array(list_double_couple_min, dtype = np.dtype('int, float'))
            abscissa_list_min = np.array(np_double_couple_min["f0"], dtype = int) # список значений абсциссы
            ordinata_list_min = np.array(np_double_couple_min['f1'], dtype = float) # список значений ординаты
            A = np.vander(abscissa_list_min, 2)
            coeff_min, sse_min, rank_min , sing_a_min = np.linalg.lstsq(A,ordinata_list_min, rcond=-1)
#            coeff_min = [angle_2, b_coeff_2]
    #       coeff = [a,b] где coeff[0] - это тангенс угла наклона, а coeff[1] - +b 
            list_kasanie_min = [] # временный список значений абсцыссы, в которых касательная касается графика в 3 и более точках
            list_kasanie_min_broken = []
            dots_above_line_min = 0 # вышли за пределы линии
            dots_under_line_min = 0
            for i in min_listx: # сбор касаний я начинаю с первой точки
                if (float(coeff_min[0] * i) + float(coeff_min[1]) - price(i)) > zazor: 
                    dots_above_line_min += 1
                    list_kasanie_min = []
                elif (price(i) - (float(coeff_min[0] * i) + float(coeff_min[1]))) > zazor:
                    dots_under_line_min += 1
                else:
                    list_kasanie_min.append(i)
                    list_kasanie_min_broken.append(i)
            list_kasanie_min.append(end_dot_3)       
            list_kasanie_min_broken.append(end_dot_3)          
            if (list_kasanie_min[-1] - list_kasanie_min[0]) > Lmin and len(list_kasanie_min) > 4: #тренд захватывает последнюю точку и длиннее минимума
                list_of_lists_min_3andmore_kasanie.append(list_kasanie_min)
                if list_kasanie_min_broken != list_kasanie_min and (dots_under_line_min/dots_above_line_min) > deep_of_trend_brokiness:
                    list_of_lists_min_3andmore_kasanie_broken.append(list_kasanie_min_broken)    
    ##########################################################################################################  
    ##__несломанные линии поддержки___#############################################
        min_set = set(tuple(x) for x in list_of_lists_min_3andmore_kasanie)
        min_list_abscissa = [list(x) for x in min_set] # это список абсцисс # индексы совпадают
        #   _max  пары (x,y) max_listx (x)
        min_list_ordinata = [] # индексы совпадают
        list_of_var_min = [] # индексы совпадают
        importance_min = [] # индексы совпадают
        importance_min_val = [] # индексы совпадают
        list_angle_min = [] # индексы совпадают
        list_sse_min = []# индексы совпадают
        real_distance_min_list = [] # индексы совпадают  
        ##__списки со списками параметров вершин и пиков, таких как высоты, основания, острота и т.д.
        tops_min_parameters = [] # индексы совпадают/ список списков [list_peaks, list_of_peaks_H, list_of_peak_W]
        peak_min_parameters = [] # индексы совпадают/ список списков [list_tops, list_of_tops_H, list_of_tops_W]
        ####____определяем уравнение тренда____####################################
        for i in min_list_abscissa: # i - список абсцисс касаний вида [x1, x2, x3..]
            d = []
            for ii in i:
                d.append(price(ii))
            min_list_ordinata.append(d)
        for i, ii in zip (min_list_abscissa, min_list_ordinata):
            list_tops = []      # список вида [(x1, y1), (x2, y2)---] с координатами вершины
            list_of_tops_H = []    # список вида [x1, x2, ...] с высотой вершины отн линии
            list_of_tops_W = [] # список вида [x1, x2, ...] с шириной основания вершины
            real_distance_min_list.append(math.hypot(i[-1] - i[0], ii[-1] - ii[0]))
            A = np.vander(i, 2)
            coeff_min, sse_min, rank_min , sing_a_min = np.linalg.lstsq(A, ii, rcond=-1)
#            coeff_min = [angle_2, b_coeff_2]
            list_angle_min.append(coeff_min)
            list_sse_min.append(sse_min)
            amount_of_peaks_on_line = len(i)
            for l in range(amount_of_peaks_on_line - 1):
                k = i[l +1]
                j = i[l]
                max_selected = [x for x in _max if (x[0] >= j and x[0] <= k)]
                if len(max_selected) > 0:# найдена нормальная вершина в этом интервале
                    max_selected_y = [x[1] for x in max_selected]
                    local_top_y = max(max_selected_y)
                    list_of_tops_W.append((k-j))
                    local_top = [x for x in max_selected if x[1] == local_top_y][0]
                    list_tops.append(local_top)
                    local_top_H = local_top_y - coeff_min[0]*local_top[0] - coeff_min[1]
                    list_of_tops_H.append(local_top_H)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        list_of_tops_H.insert(0, local_top_H)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        list_of_tops_H.append(local_top_H)
                else: # вершина в интервале не найдена. будем использовать середину интервала
                    max_selected = int((k + j)/2)
                    local_top_y = price(max_selected)
                    list_of_tops_W.append((k-j))
                    local_top = (max_selected, local_top_y)
                    list_tops.append(local_top)
                    local_top_H = local_top_y - coeff_min[0]*local_top[0] - coeff_min[1]
                    list_of_tops_H.append(local_top_H)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        list_of_tops_H.insert(0, local_top_H)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        list_of_tops_H.append(local_top_H)
            tops_sharpness = [(x/y) for x,y in zip(list_of_tops_H, list_of_tops_W)] 
            tops_min_parameters.append([list_tops, list_of_tops_H, list_of_tops_W, tops_sharpness])
        for i, j, k in zip(min_list_abscissa, tops_min_parameters, min_list_ordinata): #между образующими тренд пиками есть по 1 вершине
            list_peaks = [(x, y) for x,y in zip(i, k)]      # список вида [(x1, y1), (x2, y2)---] с координатами пиков
            list_of_peak_H = [] # список вида [x1, x2, ...] с высотой пиков отн линии соседних вершин
            list_of_peak_W = [] # список вида [x1, x2, ...] с шириной пиков по соседним вершинам  
            list_peaks = [(x, y) for x,y in zip(i, k)]
            jj = j[0]
            for l in range(len(i)):
                peak_x = i[l]
                peak_y = k[l]
                first_neighbour_top = jj[l]
                second_neighbour_top = jj[l + 1]
                peak_W = second_neighbour_top[0] - first_neighbour_top[0]
                tg_angle = (second_neighbour_top[1] - first_neighbour_top[1])/(peak_W)
                b_coeff = first_neighbour_top[1] - tg_angle*first_neighbour_top[0]
                peak_H = tg_angle*peak_x + b_coeff - peak_y
                list_of_peak_W.append(peak_W)
                list_of_peak_H.append(peak_H)
            peak_sharpness = [(x/y) for x,y in zip(list_of_peak_H, list_of_peak_W)] 
            peak_min_parameters.append([list_peaks, list_of_peak_H, list_of_peak_W, peak_sharpness])
        for i in min_list_abscissa:
            c =  [x-x2 for x, x2 in zip(i[1:], i[:-1])]
            mean_distance = np.mean(c)
            cc = [(abs(x - mean_distance)) for x in c]
            list_of_var_min.append(np.mean(cc))
        for i1, i2, i5, i6 in zip(min_list_abscissa, list_of_var_min, real_distance_min_list, list_sse_min):
            importance_min.append(importance_of_line((len(i1)), i5, i2))
            importance_min_val.append([i1, i2, i5, i6])
    ##__сломанные линии поддержки___###############################################
        min_set_broken = set(tuple(x) for x in list_of_lists_min_3andmore_kasanie_broken)
        min_list_abscissa_broken = [list(x) for x in min_set_broken] # это список абсцисс # индексы совпадают
        min_list_ordinata_broken = [] # индексы совпадают
        list_of_var_min_broken = [] # индексы совпадают
        importance_min_broken = [] # индексы совпадают
        importance_min_val_broken = [] # индексы совпадают
        list_angle_min_broken = [] # индексы совпадают
        list_sse_min_broken = []# индексы совпадают
        ##__списки со списками параметров вершин и пиков, таких как высоты, основания, острота и т.д.
        tops_min_parameters_broken = [] # индексы совпадают/ список списков [list_peaks, list_of_peaks_H, list_of_peak_W]
        peak_min_parameters_broken = [] # индексы совпадают/ список списков [list_tops, list_of_tops_H, list_of_tops_W]
        integrals_of_broken_parts = [] # индексы совпадают/ список списков интегралов вышедших частей
        broken_parts_H = [] # индексы совпадают/ список списков высоты вышедших частей
        persentage_of_brokeness_min = [] # индексы совпадают/ отношение площади всех пробоев к площади вершин
        for i in min_list_abscissa_broken:
            d = []
            for ii in i:
                d.append(price(ii))
            min_list_ordinata_broken.append(d)
        real_distance_min_list_broken = []    
        for i, ii in zip (min_list_abscissa_broken, min_list_ordinata_broken):
            list_tops = []      # список вида [(x1, y1), (x2, y2)---] с координатами вершины
            list_of_tops_H = []    # список вида [x1, x2, ...] с высотой вершины отн линии
            list_of_tops_W = [] # список вида [x1, x2, ...] с шириной основания вершины
            real_distance_min_list_broken.append(math.hypot(i[-1] - i[0], ii[-1] - ii[0]))
            A = np.vander(i, 2)
            coeff_min, sse_min, rank_min , sing_a_min = np.linalg.lstsq(A, ii, rcond=-1)
#            coeff_min = [angle_2, b_coeff_2]
            list_angle_min_broken.append(coeff_min)
            list_sse_min_broken.append(sse_min)
            amount_of_peaks_on_line = len(i)
            for l in range(amount_of_peaks_on_line - 1):
                k = i[l +1]
                j = i[l]
                max_selected = [x for x in _max if (x[0] >= j and x[0] <= k)]
                if len(max_selected) > 0:# найдена нормальная вершина в этом интервале
                    max_selected_y = [x[1] for x in max_selected]
                    local_top_y = max(max_selected_y)
                    list_of_tops_W.append((k-j))
                    local_top = [x for x in max_selected if x[1] == local_top_y][0]
                    list_tops.append(local_top)
                    local_top_H = local_top_y - coeff_min[0]*local_top[0] - coeff_min[1]
                    if local_top_H >= 0:
                        list_of_tops_H.append(local_top_H)
                    else:
                        list_of_tops_H.append(0)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.insert(0, local_top_H)
                        else:
                            list_of_tops_H.insert(0, 0)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.append(local_top_H)
                        else:
                            list_of_tops_H.append(0)
                else: # вершина в интервале не найдена. будем использовать середину интервала
                    max_selected = int((k + j)/2)
                    local_top_y = price(max_selected)
                    list_of_tops_W.append((k-j))
                    local_top = (max_selected, local_top_y)
                    list_tops.append(local_top)
                    local_top_H = local_top_y - coeff_min[0]*local_top[0] - coeff_min[1]
                    if local_top_H >= 0:
                        list_of_tops_H.append(local_top_H)
                    else:
                        list_of_tops_H.append(0)
                    ## выдумываем первую и последнюю вершину 
                    if l == 0:
                        x_of_first_top = j - (k - j)
                        local_top = (x_of_first_top, local_top_y)
                        list_tops.insert(0,local_top)
                        list_of_tops_W.insert(0, (k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.insert(0, local_top_H)
                        else:    
                            list_of_tops_H.insert(0, 0)
                        list_of_tops_H.insert(0, local_top_H)
                    if l == amount_of_peaks_on_line - 2:
                        x_of_last_top = k + (k - j)
                        local_top = (x_of_last_top, local_top_y)
                        list_tops.append(local_top)
                        list_of_tops_W.append((k-j))
                        if local_top_H >= 0:
                            list_of_tops_H.append(local_top_H)
                        else:    
                            list_of_tops_H.append(0)
            tops_sharpness = [(x/y) for x,y in zip(list_of_tops_H, list_of_tops_W)] 
            tops_integral_rough = [(x*y/2) for x,y in zip(list_of_tops_H, list_of_tops_W)]
            tops_min_parameters_broken.append([list_tops, list_of_tops_H, list_of_tops_W, tops_sharpness, tops_integral_rough])
        for i, j, k in zip(min_list_abscissa_broken, tops_min_parameters_broken, min_list_ordinata_broken): #между образующими тренд пиками есть по 1 вершине
            list_peaks = [(x, y) for x,y in zip(i, k)]      # список вида [(x1, y1), (x2, y2)---] с координатами пиков
            list_of_peak_H = [] # список вида [x1, x2, ...] с высотой пиков отн линии соседних вершин
            list_of_peak_W = [] # список вида [x1, x2, ...] с шириной пиков по соседним вершинам  
            list_peaks = [(x, y) for x,y in zip(i, k)]
            jj = j[0]
            for l in range(len(i)):
                peak_x = i[l]
                peak_y = k[l]
                first_neighbour_top = jj[l]
                second_neighbour_top = jj[l + 1]
                peak_W = second_neighbour_top[0] - first_neighbour_top[0]
                tg_angle = (second_neighbour_top[1] - first_neighbour_top[1])/(peak_W)
                b_coeff = first_neighbour_top[1] - tg_angle*first_neighbour_top[0]
                peak_H = tg_angle*peak_x + b_coeff - peak_y
                list_of_peak_W.append(peak_W)
                list_of_peak_H.append(peak_H)
            peak_sharpness = [(x/y) for x,y in zip(list_of_peak_H, list_of_peak_W)] 
            peak_min_parameters_broken.append([list_peaks, list_of_peak_H, list_of_peak_W, peak_sharpness])
        for i in min_list_abscissa_broken:
            c =  [x-x2 for x, x2 in zip(i[1:], i[:-1])]
            mean_distance = np.mean(c)
            с_broken = [(abs(x - mean_distance)) for x in c]
            list_of_var_min_broken.append(np.mean(с_broken))
            list_of_proboys_min = [] # список значений x, где график пересекает тренд
            list_of_proboys_min_intervals = []
            first_dot = i[0]
            last_dot = i[-1] + 1
            if ((float(coeff_min[0] * last_dot) + float(coeff_min[1])) - zazor) < price(last_dot): # только если последняя точка в границах тренда:
                if ((float(coeff_min[0] * first_dot) + float(coeff_min[1])) - zazor) < price(first_dot):
                    marker_0 = 0 # первая точка в границах тренда 
                    marker_00 = 0
                else:
                    marker_0 = 1 # первая точка уехала за границы тренда
                    marker_00 = 1
                for ii in x_list[first_dot + 1:last_dot]:
                    marker_1 = marker_0
                    if ((float(coeff_min[0] * ii) + float(coeff_min[1])) - zazor) < price(ii):
                        marker_0 = 0
                    else:
                        marker_0 = 1
                    if marker_0 - marker_1 == 1: # 
                        list_of_proboys_min.append(ii)
                    if marker_0 - marker_1 == -1: # 
                        list_of_proboys_min.append(ii)
                list_of_proboys_min_val = len(list_of_proboys_min)        
                if marker_00 == 0 and list_of_proboys_min_val % 2 == 0: 
                   for i,k in zip(list_of_proboys_min[0::2], list_of_proboys_min[1::2]):
                       list_of_proboys_min_intervals.append([i,k])
                if marker_00 == 1 and list_of_proboys_min_val % 2 != 0: 
                   for i,k in zip(list_of_proboys_min[1::2], list_of_proboys_min[2::2]):
                       list_of_proboys_min_intervals.append([i,k])
            list_of_H_min = [] # высоты пробоев
            list_of_S_min = [] # площади пробоев
            list_of_SH_list_min = []
            list_of_W_min = []          
            for i in list_of_proboys_min_intervals:
                start = i[0]
                end = i[1] + 1
                local_w = (end - start -1)
                list_of_W_min.append(local_w)
                local_min_list = []
                for ii in x_list[start: end]:
                    local_min_list.append((float(coeff_min[0] * ii) + float(coeff_min[1])) - zazor - price(ii))
                list_of_SH_list_min.append(local_min_list)
            list_of_S_min = [sum(x) for x in list_of_SH_list_min] 
            list_of_H_min = [max(x) for x in list_of_SH_list_min]
            integrals_of_broken_parts.append(list_of_S_min)
            broken_parts_H.append(list_of_H_min)
        for i, j in zip(tops_min_parameters_broken, integrals_of_broken_parts):
            ii = i[-1]
            s2 = sum(ii)
            s1 = sum(j)
            s = s1/s2
            persentage_of_brokeness_min.append(s)
        for i1, i2, i5, i6 in zip(min_list_abscissa_broken, list_of_var_min_broken, real_distance_min_list_broken, list_sse_min):
            importance_min_broken.append(importance_of_line((len(i1)), i5, i2))
            importance_min_val_broken.append([i1, i2, i5, i6])                    
    ###############################################################################
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
                the_best_min_index_value = the_best_min_index[0]
                final_list_lower.append([fb, breakout, the_best_min, list_angle_min[the_best_min_index_value], importance_min_val[the_best_min_index_value],  tops_min_parameters[the_best_min_index_value], len(tops_min_parameters[the_best_min_index_value][0]), peak_min_parameters[the_best_min_index_value], len(peak_min_parameters[the_best_min_index_value][0])])
                ##################################################################
##                win = lb - fb
##                past_window = lb -int(win/2)
##                future_window = lb +int(win/10)
##                
##                xplot = x_list[(past_window):(future_window)]
##                yplot = y_list[(past_window):(future_window)] 
##                xx = x_list[(past_window) : (future_window)]
##                boarder_x1 = [lb, lb +1]
##                boarder_y1 = [max(yplot), min(yplot)]
##                boarder_x2 = [fb, fb +1]
##                boarder_y2 = boarder_y1
##                line = []
##                angle = list_angle_min[the_best_min_index_value][0]
##                b_coeff = list_angle_min[the_best_min_index_value][1]
##                for ii in xx:
##                    line.append(angle*ii + b_coeff)
##                lines = plt.plot(xplot, yplot, xx, line, boarder_x1, boarder_y1)
##                l1, l2, l3= lines
##                plt.setp(lines, linestyle='-')
##                plt.setp(l1, linewidth=1, color='b')
##                plt.setp(l2, linewidth=1, color='r')
##                plt.setp(l3, linewidth=1, color='g')
###                    plt.setp(l4, linewidth=1, color='y')
##                plt.show()
##                plt.pause(0.05)
#                ###################################################################
                historyfile = open(historyOutPath + output_normal, 'a',newline='')
                csv.writer(historyfile).writerow([2, final_list_lower])
                historyfile.close()
            else:
                b = 0
        if len(importance_max) > 0:
            a = max(importance_max)
            if a > Imin:
                the_best_max = a
                the_best_max_index = ([i for i, j in enumerate(importance_max) if j == the_best_max]) 
                the_best_max_index_value = the_best_max_index[0]
                fa = max_list_abscissa[the_best_max_index_value][0]
                la = max_list_abscissa[the_best_max_index_value][-1]
                the_best_max_index_value = the_best_max_index[0]
                final_list_upper.append([fa, breakout, the_best_max, list_angle_max[the_best_max_index_value], importance_max_val[the_best_max_index_value],  tops_max_parameters[the_best_max_index_value], len(tops_max_parameters[the_best_max_index_value][0]), peak_max_parameters[the_best_max_index_value], len(peak_max_parameters[the_best_max_index_value][0])])

                #################################################################
##                win = la - fa
##                past_window = la -int(win/2)
##                future_window = la +int(win/10)
##                xplot = x_list[(past_window):(future_window)]
##                yplot = y_list[(past_window):(future_window)] 
##                xx = x_list[(past_window) : (future_window)]
##                boarder_x1 = [la, la +1]
##                boarder_y1 = [max(yplot), min(yplot)]
##                boarder_x2 = [fa, fa +1]
##                boarder_y2 = boarder_y1
##                line = []
##                angle = list_angle_max[the_best_max_index_value][0]
##                b_coeff = list_angle_max[the_best_max_index_value][1]
##                for ii in xx:
##                    line.append(angle*ii + b_coeff)
##                lines = plt.plot(xplot, yplot, xx, line, boarder_x1, boarder_y1)
##                l1, l2, l3 = lines
##                plt.setp(lines, linestyle='-')
##                plt.setp(l1, linewidth=1, color='b')
##                plt.setp(l2, linewidth=1, color='r')
##                plt.setp(l3, linewidth=1, color='g')
###                    plt.setp(l4, linewidth=1, color='y')
##                plt.show()
##                plt.pause(0.05)
#                ###################################################################
                historyfile = open(historyOutPath + output_normal, 'a',newline='')
                csv.writer(historyfile).writerow([1, final_list_upper])
                historyfile.close()
            else:
                a = 0  
        ###########################################################################
        final_list_lower_broken = []
        final_list_upper_broken = []
        a_broken = 0
        b_broken = 0
        la_broken = 0
        lb_broken = 0
        fa_broken = 0
        fb_broken = 0
        the_best_max_index_value_broken = 0
        the_best_min_index_value_broken = 0
        if len(importance_min_broken) > 0:
            b_broken = max(importance_min_broken)
            if b_broken > Imin:
                the_best_min_broken = b_broken
                the_best_min_index_broken = ([i for i, j in enumerate(importance_min_broken) if j == the_best_min_broken])
                the_best_min_index_value_broken = the_best_min_index_broken[0]
                fb_broken = min_list_abscissa_broken[the_best_min_index_value_broken][0]
                lb_broken = min_list_abscissa_broken[the_best_min_index_value_broken][-1]
                the_best_min_index_value_broken = the_best_min_index_broken[0]
                final_list_lower_broken.append([fb_broken, breakout, the_best_min_broken, list_angle_min_broken[the_best_min_index_value_broken], importance_min_val_broken[the_best_min_index_value_broken],  tops_min_parameters_broken[the_best_min_index_value_broken], len(tops_min_parameters_broken[the_best_min_index_value_broken][0]), peak_min_parameters_broken[the_best_min_index_value_broken], len(peak_min_parameters_broken[the_best_min_index_value_broken][0]), persentage_of_brokeness_min[the_best_min_index_value_broken]])
                #################################################################
##                win = lb_broken - fb_broken
##                past_window = lb_broken -int(win/2)
##                future_window = lb_broken +int(win/10)
##                xplot = x_list[(past_window):(future_window)]
##                yplot = y_list[(past_window):(future_window)] 
##                xx = x_list[(past_window) : (future_window)]
##                boarder_x1 = [lb_broken, lb_broken +1]
##                boarder_y1 = [max(yplot), min(yplot)]
##                boarder_x2 = [fb_broken, fb_broken +1]
##                boarder_y2 = boarder_y1
##                line = []
##                angle = list_angle_min_broken[the_best_min_index_value_broken][0]
##                b_coeff = list_angle_min_broken[the_best_min_index_value_broken][1]
##                for ii in xx:
##                    line.append(angle*ii + b_coeff)
##                lines = plt.plot(xplot, yplot, xx, line, boarder_x1, boarder_y1)
##                l1, l2, l3 = lines
##                plt.setp(lines, linestyle='-')
##                plt.setp(l1, linewidth=1, color='b')
##                plt.setp(l2, linewidth=1, color='r')
##                plt.setp(l3, linewidth=1, color='g')
###                    plt.setp(l4, linewidth=1, color='y')
##                plt.show()
##                plt.pause(0.05)
#                ###################################################################
                historyfile = open(historyOutPath + output_broken, 'a',newline='')
                csv.writer(historyfile).writerow([2, final_list_lower_broken])
                historyfile.close()
            else:
                b_broken = 0
        if len(importance_max_broken) > 0:
            a_broken = max(importance_max_broken)
            if a_broken > Imin:
                the_best_max_broken = a_broken
                the_best_max_index_broken = ([i for i, j in enumerate(importance_max_broken) if j == the_best_max_broken]) 
                the_best_max_index_value_broken = the_best_max_index_broken[0]
                fa_broken = max_list_abscissa_broken[the_best_max_index_value_broken][0]
                la_broken = max_list_abscissa_broken[the_best_max_index_value_broken][-1]
                the_best_max_index_value_broken = the_best_max_index_broken[0]
                final_list_upper_broken.append([fa_broken, breakout, the_best_max_broken, list_angle_max_broken[the_best_max_index_value_broken], importance_max_val_broken[the_best_max_index_value_broken],  tops_max_parameters_broken[the_best_max_index_value_broken], len(tops_max_parameters_broken[the_best_max_index_value_broken][0]), peak_max_parameters_broken[the_best_max_index_value_broken], len(peak_max_parameters_broken[the_best_max_index_value_broken][0]),  persentage_of_brokeness_max[the_best_max_index_value_broken]])
                #################################################################
##                win = la_broken - fa_broken
##                past_window = la_broken -int(win/2)
##                future_window = la_broken +int(win/10)
##                xplot = x_list[(past_window):(future_window)]
##                yplot = y_list[(past_window):(future_window)] 
##                xx = x_list[(past_window) : (future_window)]
##                boarder_x1 = [la_broken, la_broken +1]
##                boarder_y1 = [max(yplot), min(yplot)]
##                boarder_x2 = [fa_broken, fa_broken +1]
##                boarder_y2 = boarder_y1
##                line = []
##                angle = list_angle_max_broken[the_best_max_index_value_broken][0]
##                b_coeff = list_angle_max_broken[the_best_max_index_value_broken][1]
##                for ii in xx:
##                    line.append(angle*ii + b_coeff)
##                lines = plt.plot(xplot, yplot, xx, line, boarder_x1, boarder_y1)
##                l1, l2, l3 = lines
##                plt.setp(lines, linestyle='-')
##                plt.setp(l1, linewidth=1, color='b')
##                plt.setp(l2, linewidth=1, color='r')
##                plt.setp(l3, linewidth=1, color='g')
###                    plt.setp(l4, linewidth=1, color='y')
##                plt.show()
##                plt.pause(0.05)
#                ###################################################################
                historyfile = open(historyOutPath + output_broken, 'a',newline='')
                csv.writer(historyfile).writerow([1, final_list_upper_broken])
                historyfile.close()
            else:
                a_broken = 0  
