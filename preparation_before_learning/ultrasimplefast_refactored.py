# -*- coding: utf-8 -*-
"""
script №2 in order of applying
That script takes tick_data from purifier_zipifier_of_row_data, scans
it with window and writes every possible candidate to real trends in csv.file
minmax dots on plot are selected by hiperbolic distributed window grid 
(len of window is greater in far past, then in near past)
"""
import csv
import numpy as np
import pandas
inputpath='../../trends_data/preprocess/pureSBER315.csv'
historyOutPath='../../trends_data/preprocess/shit.csv'


columns = ['<TIME>', '<VOLUME>', '<PRICE>']
###############################################################################
data = pandas.read_csv(inputpath, sep = ',', names=columns, header = 0)
y_column = data['<PRICE>']
y_array = np.array(y_column)
x_list = np.array(range(len(y_array)))
ticks_total = len(y_array) - 1

step_read = 60
Lmax = 400000
zazor_coeff = 0.00005
relax_coef = 1
# сетка на 800 делений
number_of_bars = 800

# место остановки окна
pitstop = ticks_total - 1000
min_trend_len = 3000

# старт движения окна
breaker = 5000

# инициализация переменной не нулем и не единицей
direction = 456
x_stop = 0


def get_delimeters(breaker, number_of_bars, window_max_size):
    """
    :param breaker: end tick of the window
    :param number_of_bars: number of bars in
        window separating grid
    :return: a numpy array of ticks corresponding to
        hyperbolic grid for a given window
    """

    if breaker <= window_max_size:
        window_starting_tick = 10
        window_size = breaker - window_starting_tick
    else:
        window_starting_tick = breaker - window_max_size
        window_size = window_max_size

    k = window_size/np.log(number_of_bars)
    delimiters_list = []
    for i in range(number_of_bars+1):
        window_starting_tick += int(k/i)
        delimiters_list.append(window_starting_tick)

    return np.array(delimiters_list)


def get_grid_extremum_coordinates(y_array, window_delimiters):
    """
    takes arrays of ticks and corresponding prices, cuts a
    window corresponding to window_delimeters boundaries.
    Slices this 2D array according to  window_delimeters list.

    :return: dataframe with 2D arrays of ticks and prices
        for each grid section and section number as index
    """
    #TODO: code

    return sliced_window_df


def append_extremums(sliced_window_df)
    """
    takes sliced_window_df and calculates min and max 
    coordinates for each window section
    :param sliced_window_df: 
    :return: df with coordinates of max and mins
    """
    #TODO: code

    return sliced_window_extremums_df


def append_trend_equations(sliced_window_extremums_df):
    """
    adds 4 columns with trend's equation parameters for
    mins and maxs: min_k, min_b, max_k, max_b
    :param sliced_window_extremums_df:
    :return: df with 4 added columns of coefficients
    """
    #TODO: code
    return sliced_window_coef_df


def find_all_extremums_on_trends(sliced_window_coef_df, zazor_coeff):
    """
    Founds all extremums lying in zazor_coeff distance around
        every trends. Adds a column with list of coordinates
        of all extremums
    :param sliced_window_coef_df:
    :param zazor_coeff:
    :return: df with 2 new columns with coordinates of
        extremums for min and max trends
    """

    #TODO: code
    return sliced_window_kasanie


def filter_by_extremum_count_and_trend_length(sliced_window_kasanie, extremum_count, min_trend_len):
    """

    :param sliced_window_kasanie:
    :param extremum_count:
    :param min_trend_len:
    :return:
    """

    #TODO: code
    return filtered_trends_df


def extract_memorized_list(filtered_trends_df):
    """
    extracts filtered trends coordinates to required output
    :param filtered_trends_df:
    :return: 
    """
    #TODO: code

    return memorized_list


def price(some): # ввел функцию цены акции от тика, индекс начинается с единицы
    return (y_array[some])


def func(breaker, zazor, Lmax, number_of_bars):
    if breaker <= Lmax:
        window_starting_tick = 10
    elif breaker > Lmax:
        window_starting_tick = breaker - Lmax
    window_size = breaker - window_starting_tick

    a = window_size/np.log(number_of_bars)
    list_of_delimeters = []
    for i in range(number_of_bars+1):
        window_starting_tick += int(a/i)
        list_of_delimeters.append(window_starting_tick)

    list_of_delimeters = sorted(list(set(list_of_delimeters)))
    list_of_mins = np.array()
    list_of_mins_pos = np.array()
    list_of_maxs = np.array()
    list_of_maxs_pos = np.array()

    for i, j in zip(list_of_delimeters[:-1], list_of_delimeters[1:]):
        y_list_part = y_array[i:j]
        
        maxx = max(y_list_part)
        minn = min(y_list_part)
        list_of_maxs = np.append(list_of_maxs, maxx)
        list_of_mins = np.append(list_of_mins, minn)
        maxx_pos = i + np.argmax(y_list_part)
        minn_pos = i + np.argmin(y_list_part)
        list_of_maxs_pos = np.append(list_of_maxs_pos, maxx_pos)
        list_of_mins_pos = np.append(list_of_mins_pos, minn_pos)

    working_max = list_of_maxs[-1]
    working_min = list_of_mins[-1]
    working_max_pos = list_of_maxs_pos[-1]
    working_min_pos = list_of_mins_pos[-1]

    counter_max_list = list(range(len(list_of_maxs_pos)))
    ############################################################################
    # перебираем пары значений из списка максимумов

    output_set = set()

    for i, j, separ in zip(list_of_maxs[:-1], list_of_maxs_pos[:-1], counter_max_list):
        k = (working_max - i)/(working_max_pos - j)
        b = working_max - k*working_max_pos
        # временный список значений абсцыссы, в которых касательная касается графика в 4 и более точках
        list_kasanie_max = [] 
        # перебираем пары значений из списка максимумов
        for ii, jj in  zip(list_of_maxs[separ:-1], list_of_maxs_pos[separ:-1]): 
            if (ii - (k * jj + b)) > zazor:
                list_kasanie_max = []
            elif (k * jj + b - ii) > zazor:
                pass
            else:
                list_kasanie_max.append(jj)
        if len(list_kasanie_max) >= 4 and (list_kasanie_max[-1] - list_kasanie_max[0] > min_trend_len) :
            list_kasanie_max.append(working_max_pos)
            output_set.add((1, list_kasanie_max[0], list_kasanie_max[-1], k ,b, len(list_kasanie_max)))
#            csv.writer(historyfile).writerow([1, list_kasanie_max[0], list_kasanie_max[-1], k ,b, var, len(list_kasanie_max)])
    ############################################################################
    for i, j, separ in  zip(list_of_mins[:-1], list_of_mins_pos[:-1], counter_max_list): # перебираем пары значений из списка максимумов
        k = (working_min - i)/(working_min_pos - j)
        b = working_min - k*working_min_pos
        list_kasanie_min = [] # временный список значений абсцыссы, в которых касательная касается графика в 4 и более точках
        for ii, jj in  zip(list_of_mins[separ:-1], list_of_mins_pos[separ:-1]): # перебираем пары значений из списка максимумов
            if (k * jj + b - ii) > zazor:
                list_kasanie_min = []
            elif  (ii - (k * jj + b)) > zazor:
                pass
            else:
                list_kasanie_min.append(jj)
        if len(list_kasanie_min) >= 4 and (list_kasanie_min[-1] - list_kasanie_min[0] > min_trend_len):
            list_kasanie_min.append(working_min_pos)
            output_set.add((2, list_kasanie_min[0], list_kasanie_min[-1], k ,b,len(list_kasanie_min)))
#            csv.writer(historyfile).writerow([2, list_kasanie_min[0], list_kasanie_min[-1], k ,b,len(list_kasanie_min)])
    return output_set
###############################################################################
historyfile = open(historyOutPath, 'a',newline='')

memorized_set = set()

while breaker <= pitstop:
    zazor = zazor_coeff * price(breaker)
    current_set = func(breaker, zazor, historyfile, Lmax, number_of_bars)
    memorized_set.add(current_set)
    breaker += step_read
    print(breaker)

memorized_list = [list(x) for x in memorized_list]
memorized_list.sort(key=lambda x: x[1])
#######################вводим заголовок########################################
historyfile = open(historyOutPath,'a', newline='')
writer = csv.writer(historyfile, delimiter=',')
writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
writer.writerows(memorized_list)
historyfile.close()
memorized_list = [list(x) for x in memorized_list]
memorized_list.sort(key=lambda x: x[1])
#######################вводим заголовок########################################
historyfile = open(historyOutPath,'a',newline='')
writer = csv.writer(historyfile, delimiter=',')
writer.writerow(['direction', 'f_dot', 'l_dot', 'k', 'b', 'dots'])
writer.writerows(memorized_list)
historyfile.close()    