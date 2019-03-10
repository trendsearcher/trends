# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 14:11:44 2019

@author: user_PC
"""

import pandas as pd
import numpy as np
import ast

output_file_path = '../../trends_data/sber/normal_trends_outofdublers_norm2.csv'
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
# нормировка на внутренние показатели тренда (фрактальность)


def norm_by_value(row_to_normalize, normalizer):
    return [x/normalizer for x in eval(row_to_normalize)]


def line_touch_norm(line_touching_x, trend_start, trend_lenght):
    return [(x - trend_start)/trend_lenght for x in eval(line_touching_x)][1:]


def ratio_norm(ratio_col, trend_lenght, height_pic):
    return [(x*trend_lenght)/height_pic for x in eval(ratio_col)]


df2['tops_width_norm'] = df2.apply(lambda row: norm_by_value(row['tops_width'], row['trend_lenght']), axis=1)
df2['tops_height_norm'] = df2.apply(lambda row: norm_by_value(row['tops_height'], row['height_pic']), axis=1)
df2['peaks_width_norm'] = df2.apply(lambda row: norm_by_value(row['peaks_width'], row['trend_lenght']), axis=1)
df2['peaks_height_norm'] = df2.apply(lambda row: norm_by_value(row['peaks_height'], row['height_pic']), axis=1)
df2['line_touching_x_norm'] = df2.apply(lambda row: line_touch_norm(row['line_touching_x'], row['trend_start'], row['trend_lenght']), axis=1)

df2['peaks_HW_ratio_norm'] = df2.apply(lambda row: ratio_norm(row['peaks_HW_ratio'], row['trend_lenght'], row['height_pic']), axis=1)
df2['tops_HW_ratio_norm'] = df2.apply(lambda row: ratio_norm(row['tops_HW_ratio'], row['trend_lenght'], row['height_pic']), axis=1)
df2['trend_lenght_high_ratio'] = df2['height_pic']/df2['trend_lenght']

extremum_stats_columns = ['tops_width_norm', 'tops_height_norm', 
                          'peaks_width_norm', 'peaks_height_norm', 
                          'line_touching_x_norm', 'peaks_HW_ratio_norm']

for col in extremum_stats_columns:
    df[col.replace('norm', 'std')]= df[col].apply(np.std)
    df[col.replace('norm', 'mean')]= df[col].apply(np.mean)
    df[col.replace('norm', 'meadian')]= df[col].apply(np.median)


sLength = len(df2['trend_start'])

to_drop_list = ['line_touching_x', 'peaks_coordinates',
                'tops_coordinates', 'dispersion',
                'tops_height', 'tops_width',
                'peaks_width', 'peaks_height',
                'peaks_HW_ratio', 'tops_HW_ratio',
                'trend_lenght' ,'height_pic']

df2.drop(columns=to_drop_list, inplace=True)

print(df2.info())  
  
df2.to_csv(output_file_path, index=False)    