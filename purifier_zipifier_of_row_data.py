 # -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""
import csv
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from operator import itemgetter
import os
import pandas
import pylab
colnames = ['<DATE>',' <TIME>',' <BID>',' <ASK>',' <LAST>',' <VOLUME>']
data = pandas.read_csv('C:\\Users\\user_PC\\Desktop\\ugly\\SBRF.csv', names=colnames, sep = '\t', header = 1, skip_blank_lines=True)
cols = data.columns
y_maxx = data.fillna(method = 'ffill')[cols[4]]
y_lismaxx = list(y_maxx)
y_minn = data.fillna(method = 'ffill')[cols[3]]
y_lisminn = list(y_minn)
y_list = [(x+y)/200 for x,y in (zip(y_lismaxx, y_lisminn)) if (x != 0 and y != 0)] 
ticks_total = len(y_list) - 1 
print(ticks_total)
## удаляем неинформативные точки
purelisty = []
for i in range(len(y_list)):
    if i  < ticks_total:
        if y_list[i] == y_list[i + 1]:
            pass
        else:
            purelisty.append(y_list[i])
y_list = purelisty
x_list = list(range(len(y_list)))
ticks_total = len(y_list) - 1


purelisty = []
for i in range(len(y_list)):
    if i  < ticks_total:
        if y_list[i] == y_list[i + 1]:
            pass
        else:
            purelisty.append(y_list[i])
y_list = purelisty
print(len(y_list))




historyfile = open("C:\\Users\\user_PC\\Desktop\\ugly\\SBRFpure.csv",'a',newline='')
writer = csv.writer(historyfile, delimiter=',')
writer.writerow(['<price_eba>'])

for i in y_list:
    writer.writerow([i])
historyfile.close()
#
#
#























     