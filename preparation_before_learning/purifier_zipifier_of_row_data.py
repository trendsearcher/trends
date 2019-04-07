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
data = pd.read_csv('C:\\Users\\user\\Desktop\\hist\\SBRF-6.16.csv', names=colnames, sep = '\t', header = 0, skip_blank_lines=True)
data[' <BID>'].fillna(0, inplace=True)
data[' <ASK>'].fillna(0, inplace=True)
data[' <LAST>'].fillna(0, inplace=True)
data[' <VOLUME>'].fillna(0, inplace=True)

#data = np.array_split(data, 5000)[0]
#print(data)
cols = data.columns

start_of_session = datetime.strptime('10:00:00.000', '%H:%M:%S.%f')
end_of_session = datetime.strptime('23:50:00.000', '%H:%M:%S.%f')

out_of_session_indexes = []
list_of_local_deal_time_indexes_to_delite = []
sum_of_volume__of_local_deal = 0
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
        
data['<PRICE>'] = pd.Series(list_of_prices, index=data.index)
data.drop(columns=['<DATE>'], inplace=True)
data.drop(columns=[' <BID>'], inplace=True)
data.drop(columns=[' <ASK>'], inplace=True) 
data.drop(columns=[' <LAST>'], inplace=True) 
list_of_indexes_to_drop = list(set(out_of_session_indexes)) 
data.drop(data.index[out_of_session_indexes], inplace = True)
data['<PRICE>'].fillna(method = 'ffill', inplace=True)
data.to_csv('C:\\Users\\user\\Desktop\\pureSBER616.csv', index=False) 
cols = data.columns
print(cols)  
print('---')
print(data)
#
#
#
#
#  
