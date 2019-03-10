# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:11:44 2019

@author: user_PC
"""
import pandas as pd
import numpy as np
import ast

output_file_path = '../../trends_data/sber/normal_trends_outofdublers_norm.csv'
input_file_path = '../../trends_data/sber/normal_trends.csv'

df = pd.read_csv(input_file_path, header= 0, error_bad_lines=False)
df=df.drop_duplicates(subset=['trend_start', 'trend_end'], keep='first')
df = df[df["peaks_count"] < 10]
#удаление близко стоящих трендов
df = df.sort_values(by=['trend_end'])
df["old_trend_start"] = df["trend_start"].shift(1)
df["old_trend_end"] = df["trend_end"].shift(1)
df.replace(to_replace=np.nan, value=0, inplace=True)
df["min_difference_old"] = (df["old_trend_end"]  - df["old_trend_start"])
df["min_difference_new"] = (df["trend_end"]  - df["trend_start"])
df.loc[(df["min_difference_new"] > df["min_difference_old"])]
df["min_difference1"] = df.loc[(df["min_difference_new"] < df["min_difference_old"])]["min_difference_old"]
df["min_difference2"] = df.loc[(df["min_difference_new"] >= df["min_difference_old"])]["min_difference_new"]
df.replace(to_replace=np.nan, value=0, inplace=True)
df["min_difference"] = df["min_difference1"] + df["min_difference2"]
df["neighbour"] =(abs(df["old_trend_start"] - df["trend_start"]) + abs(df["old_trend_end"] - df["trend_end"])) > df["min_difference"]/2
df.drop(df[df.neighbour == False].index, inplace=True)
df.drop(columns=['old_trend_start', 'old_trend_end', "neighbour", 'min_difference_old', 'min_difference_new', 'min_difference1', 'min_difference2', 'min_difference'], inplace=True)
###########################################
'''['direction', 'trend_start', 'trend_end', 'importance', 'k', 'b', 'line_touching_x','dispersion', 'trend_lenght', 'r_squared_of_trend', 'tops_coordinates','tops_height','tops_width','tops_HW_ratio','tops_count','peaks_coordinates','peaks_height','peaks_width','peaks_HW_ratio','peaks_count', 'height_pic'''



df2 = df
#trend_touches_list = []
trend_touching_list_mean = []
trend_touching_list_std =[] 
tops_height_list_std = []
tops_height_list_mean = []
peaks_width_list_std = []
peaks_width_list_mean = []
tops_width_list_std = []
tops_width_list_mean = []
peaks_height_list_std =[]
peaks_height_list_mean = []
peaks_HW_ratio_list_std = []
peaks_HW_ratio_list_mean = []
tops_HW_ratio_list_std = []
tops_HW_ratio_list_mean = []

trend_touching_list_median = []
tops_height_list_median = []
peaks_width_list_median = []
tops_width_list_median = []
peaks_height_list_median = []
peaks_HW_ratio_list_median = []
tops_HW_ratio_list_median = []
trend_lenght_high_ratio_list = []
# нормировка на внутренние показатели тренда (фрактальность)
for index, row in df.iterrows():
    height_pic = row['height_pic']
    trend_start = row['trend_start']
    trend_lenght = row['trend_lenght']
    trend_lenght_high_ratio_list.append(height_pic/trend_lenght)
    line_touching_x = ast.literal_eval(row['line_touching_x'])
    tops_height = ast.literal_eval(row['tops_height'])
    peaks_width = ast.literal_eval(row['peaks_width'])
    peaks_height = ast.literal_eval(row['peaks_width'])
    tops_width = ast.literal_eval(row['tops_width'])
    peaks_HW_ratio = ast.literal_eval(row['peaks_HW_ratio'])
    tops_HW_ratio = ast.literal_eval(row['tops_HW_ratio'])
    tops_width_norm = [x/trend_lenght for x in tops_width]
    tops_height_norm = [x/height_pic for x in tops_height]
    peaks_width_norm = [x/trend_lenght for x in peaks_width]
    line_touching_x_norm = [(x - trend_start)/trend_lenght for x in line_touching_x][1:]
    peaks_height_norm = [x/height_pic for x in peaks_height]
    peaks_HW_ratio_norm = [x*trend_lenght/height_pic for x in peaks_HW_ratio]
    tops_HW_ratio_norm = [x*trend_lenght/height_pic for x in tops_HW_ratio]
  
    
    tops_height_norm_std = np.std(tops_height_norm)
    tops_height_list_std.append(tops_height_norm_std)
    tops_height_norm_mean = np.mean(tops_height_norm)
    tops_height_list_mean.append(tops_height_norm_mean)
    peaks_width_norm_std = np.std(peaks_width_norm)
    peaks_width_list_std.append(peaks_width_norm_std)
    peaks_width_norm_mean = np.mean(peaks_width_norm)
    peaks_width_list_mean.append(peaks_width_norm_mean)
    tops_width_list_std.append(np.std(tops_width_norm))
    tops_width_list_mean.append(np.mean(tops_width_norm))
    mean_touching_x = np.mean(line_touching_x_norm)
    std_trend_touching = np.std(line_touching_x_norm)
    trend_touching_list_mean.append(mean_touching_x)
    trend_touching_list_std.append(std_trend_touching)   
    peaks_height_list_std.append(np.std(peaks_height_norm))
    peaks_height_list_mean.append(np.mean(peaks_height_norm))
    peaks_HW_ratio_list_std.append(np.std(peaks_HW_ratio_norm))
    peaks_HW_ratio_list_mean.append(np.mean(peaks_HW_ratio_norm))
    tops_HW_ratio_list_std.append(np.std(tops_HW_ratio_norm))
    tops_HW_ratio_list_mean.append(np.mean(tops_HW_ratio_norm))
    
    trend_touching_list_median.append(np.median(line_touching_x_norm))
    tops_height_list_median.append(np.median(tops_height_norm))
    peaks_width_list_median.append(np.median(peaks_width_norm))
    tops_width_list_median.append(np.median(tops_width_norm))
    peaks_height_list_median .append(np.median(peaks_height_norm))
    peaks_HW_ratio_list_median .append(np.median(peaks_HW_ratio_norm))
    tops_HW_ratio_list_median .append(np.median(tops_HW_ratio_norm))
    

sLength = len(df2['trend_start'])
df2['trend_touching_std'] = pd.Series(trend_touching_list_std, index=df2.index)  
df2['trend_touching_mean'] = pd.Series(trend_touching_list_mean, index=df2.index)
df2['trend_touching_median'] = pd.Series(trend_touching_list_median, index=df2.index)
df2['tops_height_std'] = pd.Series(tops_height_list_std, index=df2.index) 
df2['tops_height_mean'] = pd.Series(tops_height_list_mean, index=df2.index) 
df2['tops_height_median'] = pd.Series(tops_height_list_median, index=df2.index) 
df2['peaks_width_std'] = pd.Series(peaks_width_list_std, index=df2.index) 
df2['peaks_width_mean'] = pd.Series(peaks_width_list_mean, index=df2.index) 
df2['peaks_width_median'] = pd.Series(peaks_width_list_median, index=df2.index) 
df2['tops_width_std'] = pd.Series(tops_width_list_std, index=df2.index) 
df2['tops_width_mean'] = pd.Series(tops_width_list_mean, index=df2.index) 
df2['tops_width_median'] = pd.Series(tops_width_list_median, index=df2.index) 
df2['peaks_height_std'] = pd.Series(peaks_height_list_std, index=df2.index) 
df2['peaks_height_mean'] = pd.Series(peaks_height_list_mean, index=df2.index) 
df2['peaks_height_median'] = pd.Series(peaks_height_list_median, index=df2.index) 
df2['tops_HW_ratio_std'] = pd.Series(tops_HW_ratio_list_std, index=df2.index) 
df2['tops_HW_ratio_mean'] = pd.Series(tops_HW_ratio_list_mean, index=df2.index) 
df2['tops_HW_ratio_median'] = pd.Series(tops_HW_ratio_list_mean, index=df2.index) 
df2['trend_lenght_high_ratio'] = pd.Series(trend_lenght_high_ratio_list, index=df2.index) 


df2.drop(columns=['line_touching_x'], inplace=True)
df2.drop(columns=['peaks_coordinates'], inplace=True)
df2.drop(columns=['tops_coordinates'], inplace=True)
df2.drop(columns=['dispersion'], inplace=True)
df2.drop(columns=['tops_height'], inplace=True)
df2.drop(columns=['tops_width'], inplace=True)
df2.drop(columns=['peaks_width'], inplace=True)
df2.drop(columns=['peaks_height'], inplace=True)
df2.drop(columns=['peaks_HW_ratio'], inplace=True)
df2.drop(columns=['tops_HW_ratio'], inplace=True)
df2.drop(columns=['trend_lenght'], inplace=True)
df2.drop(columns=['height_pic'], inplace=True)
trend_lenght

print(df2.info())  
  
df2.to_csv(output_file_path, index=False)    