# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 22:50:12 2018

@author: user_PC
"""
import csv
import matplotlib.pyplot as plt
import pandas

''' [1, 2, 3, 4, 5] оценка тренда по опыту'''
''' [1, 2, 3]  оценка тренда по факту пересечения в окне будущего (факт)'''

datapath='C:\\Users\\user_PC\\Desktop\\ugly\\SBRFpure.csv'
trendspath="C:\\Users\\user_PC\\Desktop\\good_bad\\normal_trends2.csv"
marked_normal_trendspath = "C:\\Users\\user_PC\\Desktop\\good_bad\\marked_normal_trendspath.csv"
colnames = ['<price_eba>']
counter_list = []
'''         header = 1  !!!    '''
data = pandas.read_csv(datapath, names=colnames, sep = '\t', header = 1)
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
        angle = float(basic_parameters[3].split('([')[1])
        b_coeff = float(basic_parameters[4].split('])')[0])
        if I > 15 and length > 5000:
            if (direction_of_deal == 1 and angle <= 0) or (direction_of_deal == 2 and angle >=0):
                counter_list.append([first_dot, last_dot, angle, b_coeff, i])
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
    last_dot = my_list[i][1]
    angle = my_list[i][2]
    b_coeff = my_list[i][3]
    future_window = int((last_dot - first_dot)/1.3) + last_dot
    future_window2 = int((last_dot - first_dot)) + last_dot
    if (first_dot - int((last_dot - first_dot)/2)) < 1000:
        past_window = 1000
    else:
        past_window = first_dot - int((last_dot - first_dot)/2)
        past_window2 = first_dot - int((last_dot - first_dot))
    xplot = x_list[(past_window):(future_window)]
    yplot = y_list[(past_window):(future_window)] 
    xx = x_list[(past_window) : (future_window)]
    boarder_x1 = [last_dot, last_dot +1]
    boarder_y1 = [max(yplot), min(yplot)]
    boarder_x2 = [first_dot, first_dot +1]
    boarder_y2 = [max(yplot), min(yplot)]
    line = []
    for ii in xx:
        line.append(angle*ii + b_coeff)
    lines = plt.plot(xplot, yplot, xx, line, boarder_x1, boarder_y1, boarder_x2, boarder_y2)
    l1, l2, l3, l4 = lines
#    plt.setp(lines, linestyle='-')
#    plt.setp(l1, linewidth=1, color='b')
#    plt.setp(l2, linewidth=1, color='r')
#    plt.setp(l3, linewidth=1, color='g')
#    plt.setp(l4, linewidth=1, color='y')
#    plt.show()
#    plt.pause(0.05)
    #####################################
    var = input("1.2 - оценки \\ число - стираем ошибку \\ Nan - записываем: ")
    if len(var) == 3:
        value = [int(x) for x in list(var.split('.'))]
        marked_list = (my_list[i][-1])
        marked_list.append(value)
        GOD_LIST.append(marked_list)#
        i += 1
    elif len(var) == 1:
        del GOD_LIST[-1]
        i -= 1
    elif len(var) == 0:
        print("записали")
        for j in GOD_LIST:
            if len(j) == 4:
                del j[-2]
            historyfile = open(marked_normal_trendspath, 'a',newline='')
            csv.writer(historyfile).writerow(j)
            historyfile.close()
        i += 0  
    else:
        print('НЕ ТО ВВЕЛ!')
