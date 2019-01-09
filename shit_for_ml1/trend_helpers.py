import pandas as pd
import numpy as np
import os
import itertools

def superlist_parser(trend):
    """
    берет формат записи параметров трендов курильщика и 
    переводит его в формат записи нормального человека (датафрейм)
    """
    trend=trend.replace('array([], dtype=float64)', 'array([0])')
    trend=trend.replace('array', 'np.array')
    trend_list=eval(trend)
    trend_list=trend_list[0]
    #extract parameters
    trend_start=trend_list[0]
    trend_end=trend_list[1]
    importance=trend_list[2]
    k=trend_list[3][0]
    b=trend_list[3][1]
    dispersion=trend_list[4][1]
    absolute_trend_lenght=trend_list[4][2]
    r_squared_of_trend=trend_list[4][3][0] #да бля тут аррей из одной цифры!
    tops_coordinates=trend_list[5][0]
    tops_height=trend_list[5][1]
    tops_width=trend_list[5][2]
    tops_HW_ratio=trend_list[5][3]
    tops_count=trend_list[6]
    peaks_coordinates=trend_list[7][0]
    peaks_height=trend_list[7][1]
    peaks_width=trend_list[7][2]
    peaks_HW_ratio=trend_list[7][3]
    peaks_count=trend_list[8]
    return trend_start,trend_end,importance,k,b,dispersion,absolute_trend_lenght,r_squared_of_trend,tops_coordinates,tops_height, tops_width, tops_HW_ratio, tops_count,peaks_coordinates, peaks_height, peaks_width, peaks_HW_ratio, peaks_count 

def pretify_dataframe(df):
    """
    используя предыдущую функцию собираем красивый датафрейм
    где каждая строка - тренд. 
    """
    df['trend_start'],df['trend_end'],df['importance'],df['k'],df['b'],df['dispersion'],df['absolute_trend_lenght'],df['r_squared_of_trend'],df['tops_coordinates'],df['tops_height'],df['tops_width'],df['tops_HW_ratio'],df['tops_count'],df['peaks_coordinates'],df['peaks_height'],df['peaks_width'],df['peaks_HW_ratio'],df['peaks_count']=zip(*df["trend_params"].map(superlist_parser))
    #распарсили формат курилщика, больше эта колонка не нужна, выкидываем
    df=df.drop('trend_params',  axis=1)
    return df

def extremum_spliter(df):
    """
    функция режет каждый тренд на вершины и пики и возвращает
    новый датафрейм. Теперь каждая строка - 
    пик или вершинка тренда. Информация о принадлеждности
    пика к определенному тренду сохраняется с помощью
    колонки ID
    """
    #TODO
    return None