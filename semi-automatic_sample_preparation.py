# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:50:12 2018

@author: user_PC
"""
import csv
import matplotlib.pyplot as plt
import pandas

''' [1, 2, 3, 4, 5] оценка тренда по опыту - насколько хорошо совпадает линия с пиками, похожи ли эти пики на пики, их распределение'''
''' [1, 2, 3, 4, 5]  прибыльность сделки (tp/sl)'''
pic2scale = 20 #регулирует масштаб второй картинки в итерации
Imin = 15 #мин важность, начиная с которой терминал показывает картинку оператору
length_min = 1000 ##мин длина тренда, начиная с которой терминал показывает картинку оператору
inputpath='C:\\Users\\user_PC\\Desktop\\rurusd2\\USDRUB_TOM_2018_07_10pure.csv'
trendspath="C:\\Users\\user_PC\\Desktop\\rurusd2\\normal_trends_crossed2.csv"
marked_normal_trendspath = "C:\\Users\\user_PC\\Desktop\\rurusd2\\marked_normal_trends_crossed2.csv"
colnames = ['<price_eba>']
counter_list = []
'''         header = 1  !!!    '''
data = pandas.read_csv(inputpath, names=colnames, sep = '\t', header = 1)
cols = data.columns
y_column = data[cols[0]]
y_list = list(y_column)
x_list = list(range(len(y_list)))
with open (trendspath , "r") as csvfile:
    reader = csv.reader(csvfile, delimiter =  ",")
    my_list0 = list(reader)
    my_list_chisto_chisto = []
    for i in my_list0:
        direction_of_deal = int(i[0])
        basic_parameters = i[1].split('[[')[1].split(',')
        first_dot = int(basic_parameters[0])
        last_dot = int(basic_parameters[1])
        length = last_dot - first_dot
        I = float(basic_parameters[2])
        angle = float(basic_parameters[3:][:-1][0][2:])
        b_coeff = float(basic_parameters[3:][:-1][1][:-1])
        sl = float(i[1].split('[[')[3].split('],')[0].split(',')[-1][:-1])# стоп лосс по последней вершины
        sl_H = float(i[1].split('[[')[3].split(']]')[0]. split('],')[1].split(',')[-1])# высота последней вершины отн тренда
        sl_dot = [int(i[1].split('[[')[3].split('],')[0].split(',')[-4][2:]), sl]
        height_of_trend = length*angle# высота тренда от первой точки до пересечения
        if I > Imin and length > length_min:
            if (direction_of_deal == 1 and angle <= 0) or (direction_of_deal == 2 and angle >=0):
                counter_list.append([first_dot, last_dot, angle, b_coeff, sl_dot])
# сортируем и удаляем дублеры        
my_list = sorted(counter_list, key = lambda  x: x[1]) 
for i in range(20):
    for i in range(len(my_list)):
        if i < len(my_list) - 1:
            if (my_list[i + 1][0] == my_list[i][0] and my_list[i + 1][1] == my_list[i][1]):
                my_list.remove(my_list[i + 1]) 
max_counter = len(my_list)
print(max_counter)
i = 0
GOD_LIST = []
while i <= max_counter:
    first_dot = my_list[i][0]
    print(i)
    st_top = my_list[i][-1]
    print(st_top)
    last_dot = my_list[i][1]
    angle = my_list[i][2]
    b_coeff = my_list[i][3]
    future_window = int((last_dot - first_dot)/1.3) + last_dot
    future_window2 = int((last_dot - st_top[0])*pic2scale) + last_dot
    if (first_dot - int((last_dot - first_dot)/2)) < 1000:
        past_window = 1000
    else:
        past_window = first_dot - int((last_dot - first_dot)/2)
        past_window2 = st_top[0] - int((last_dot - st_top[0])*pic2scale)
    xplot1 = x_list[(past_window):(future_window)]
    yplot1 = y_list[(past_window):(future_window)] 
    xx1 = x_list[(past_window) : (future_window)]
    boarder_x11 = [last_dot, last_dot +1]
    boarder_y11 = [max(yplot1), min(yplot1)]
    boarder_x21 = [first_dot, first_dot +1]
    boarder_y21 = [max(yplot1), min(yplot1)]
    sl_line_x1 = [st_top[0], xplot1[-1]]
    sl_line_y1 = [st_top[1], st_top[1]]
    line1 = []
    for ii in xx1:
        line1.append(angle*ii + b_coeff)
    lines = plt.plot(xplot1, yplot1, xx1, line1, boarder_x11, boarder_y11, boarder_x21, boarder_y21, sl_line_x1, sl_line_y1)
    l1, l2, l3, l4, l5 = lines
    plt.setp(lines, linestyle='-')
    plt.setp(l1, linewidth=1, color='b')
    plt.setp(l2, linewidth=1, color='r')
    plt.setp(l3, linewidth=1, color='g')
    plt.setp(l4, linewidth=1, color='y')
    plt.setp(l5, linewidth=1, color='b')
    plt.grid()
    plt.show()
    plt.pause(0.05)
    #######рассмотрим пересечение подробнее###################
    xplot2 = x_list[(past_window2):(future_window2)]
    yplot2 = y_list[(past_window2):(future_window2)] 
    xx2 = x_list[(past_window2) : (future_window2)]
    boarder_x12 = [last_dot, last_dot +1]
    boarder_y12 = [max(yplot2), min(yplot2)]
    sl_line_x2 = [st_top[0], xplot2[-1]]
    sl_line_y2 = [st_top[1], st_top[1]]
    line2 = []
    for ii in xx2:
        line2.append(angle*ii + b_coeff)
    lines = plt.plot(xplot2, yplot2, xx2, line2, boarder_x12, boarder_y12, sl_line_x2, sl_line_y2)
    l1, l2, l3, l4 = lines
    plt.setp(lines, linestyle='-')
    plt.setp(l1, linewidth=1, color='b')
    plt.setp(l2, linewidth=1, color='r')
    plt.setp(l3, linewidth=1, color='g')
    plt.setp(l5, linewidth=1, color='b')
    plt.grid()
    plt.show()
    plt.pause(0.05)
    print(st_top)
    print("---+++---+++---+++---+++---+++---+++---+++---+++---+++---+++---+++--")
    i+= 1
    #####################################
#    var = input("1.2 - оценки \\ число - стираем ошибку \\ Nan - записываем: ")
#    if len(var) == 3:
#        value = [int(x) for x in list(var.split('.'))]
#        marked_list = (my_list[i][-1])
#        marked_list.append(value)
#        GOD_LIST.append(marked_list)#
#        i += 1
#    elif len(var) == 1:
#        del GOD_LIST[-1]
#        i -= 1
#    elif len(var) == 0:
#        print("записали")
#        for j in GOD_LIST:
#            if len(j) == 4:
#                del j[-2]
#            historyfile = open(marked_normal_trendspath, 'a',newline='')
#            csv.writer(historyfile).writerow(j)
#            historyfile.close()
#        i += 0  
#    else:
#        print('НЕ ТО ВВЕЛ!')
