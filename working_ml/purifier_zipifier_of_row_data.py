 # -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

#import time
from datetime import datetime
import numpy as np
from operator import itemgetter
import pandas as pd



colnames = ['<DATE>',' <TIME>',' <BID>',' <ASK>',' <LAST>',' <VOLUME>']
data = pd.read_csv('C:\\Users\\user_PC\\Desktop\\rts\\RTS.csv', names=colnames, sep = '\t', header = 0, skip_blank_lines=True)
data[' <BID>'].fillna(0, inplace=True)
data[' <ASK>'].fillna(0, inplace=True)
data[' <LAST>'].fillna(0, inplace=True)
data[' <VOLUME>'].fillna(0, inplace=True)

#data = np.array_split(data, 5000)[0]

print(data)
print(data.info())
print('______________________')
cols = data.columns

start_of_session = datetime.strptime('10:00:00.000', '%H:%M:%S.%f')
end_of_session = datetime.strptime('23:50:00.000', '%H:%M:%S.%f')

out_of_session_indexes = []
list_of_local_deal_time_indexes_to_delite = []
sum_of_volume__of_local_deal = 0
sum_of_volume__of_local_deal_list = []
index_of_last_part_of_local_deal_list = []
list_of_all_deals_time = []
list_of_prices = []


time_i = 0
for i, row in data.iterrows(): 
    prev_time = time_i
    bid_i = row[' <BID>']
    ask_i = row[' <ASK>']
    last_i = row[' <LAST>']
    
    volume_i = row[' <VOLUME>']
    date_i = row['<DATE>']
    average_price = []
    if bid_i != 0:
        average_price.append(bid_i)
    if ask_i != 0:
        average_price.append(ask_i)
    if last_i != 0:
        average_price.append(last_i)
    list_of_prices.append(np.mean(average_price))
    time_i = datetime.strptime(row[' <TIME>'], '%H:%M:%S.%f') 
    
    
    if time_i > end_of_session or time_i < start_of_session:
        out_of_session_indexes.append(i)
#    if time_i == prev_time:
#        if len(list_of_local_deal_time_indexes_to_delite) == 0:
#            list_of_local_deal_time_indexes_to_delite.append(i-1)
#            list_of_local_deal_time_indexes_to_delite.append(i)
#            sum_of_volume__of_local_deal += volume_i
#        else: 
#            list_of_local_deal_time_indexes_to_delite.append(i)
#            sum_of_volume__of_local_deal += volume_i
#    if time_i != prev_time:
#        if len (list_of_local_deal_time_indexes_to_delite) != 0:
#            list_of_all_deals_time.append(list_of_local_deal_time_indexes_to_delite[:-1])
#            sum_of_volume__of_local_deal_list.append(sum_of_volume__of_local_deal)
#            sum_of_volume__of_local_deal = 0
#            index_of_last_part_of_local_deal_list.append(list_of_local_deal_time_indexes_to_delite[-1])
#            list_of_local_deal_time_indexes_to_delite = []
        
data[' <PRICE>'] = pd.Series(list_of_prices, index=data.index)
#data[' <VOLUME>'] = pd.Series(sum_of_volume__of_local_deal_list, index=index_of_last_part_of_local_deal_list) 
data.drop(columns=['<DATE>'], inplace=True)
data.drop(columns=[' <BID>'], inplace=True)
data.drop(columns=[' <ASK>'], inplace=True) 
data.drop(columns=[' <LAST>'], inplace=True) 
#list_of_indexes_to_drop = list(set(out_of_session_indexes + list_of_local_deal_time_indexes_to_delite)) 
data.drop(data.index[out_of_session_indexes], inplace = True)
data.to_csv('C:\\Users\\user_PC\\Desktop\\rts\\pureRTS18.csv', index=False) 
cols = data.columns
print(cols)  
print('---')
print(data)
#
#
#
#
#  
##y_maxx = data.fillna(method = 'ffill')[cols[2]]
##y_lismaxx = list(y_maxx)
##y_minn = data.fillna(method = 'ffill')[cols[3]]
##y_lisminn = list(y_minn)
##
##y_list = [(x+y)/2 for x,y in (zip(y_lismaxx, y_lisminn)) if (x != 0 and y != 0)] 
##ticks_total = len(y_list) - 1 
#### удаляем неинформативные точки
##purelisty = []
##for i in range(len(y_list)):
##    if i  < ticks_total:
##        if y_list[i] == y_list[i + 1]:
##            pass
##        else:
##            purelisty.append(y_list[i])
##y_list = purelisty[20:]
##x_list = list(range(len(y_list)))
##ticks_total = len(y_list) - 1
##
##plot = plt.plot(x_list,y_list)
##plt.grid()
##plt.pause(0.05) 
##
##historyfile = open("C:\\Users\\user\\Desktop\\sber2\\SBER1217_2_.csv",'a',newline='')
##writer = csv.writer(historyfile, delimiter=',')
##writer.writerow(['<price_eba>'])
##
##for i in y_list:
##    writer.writerow([i])
##historyfile.close()
#
