import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dateutil.relativedelta import *
import datetime
from scipy.signal import filtfilt, argrelextrema
from main import *


def get_df(p, file_name):
    """
    :param p: the path of excel file
    :return: a pandas dataframe containing the data we're interested in
    """
    df = pd.read_excel(os.path.join(p,file_name+SUFFIX))
    return df


def work_gg(gg_rate):
    """
    :param gg_rate: the original dataframe
    :return: modified gg_rate, dataframe to group by months, dataframe consisting of last two days,
    dataframe consisting of the second-last day, dataframe consisting of the last day only
    """
    gg_rate = gg_rate[['date', 'm12']]
    gg_m12 = gg_rate['m12']
    gg_rate['diffs'] = gg_m12.diff()

    """group by year and month"""
    gg_rate['year_month'] = gg_rate['date'].apply(lambda x: 100 * x.year + x.month)
    month_groups = gg_rate.groupby('year_month')

    """get last two days, last days and second last days"""
    last_two_days = month_groups.tail(2)
    last_two_days = last_two_days[:-1]
    second_last = last_two_days.groupby('year_month').head(1)
    last_day = month_groups.tail(1)
    first_day = month_groups.head(1)[['date', 'year_month', 'diffs']]

    return gg_rate, month_groups, last_two_days, second_last, last_day, first_day


def abs_spikes(total, last_two, second_last, last):
    """
    :param total: the dataframe consisting all information
    :param last_two: dataframe consisting of last two days
    :param second_last: dataframe consisting of the second-last day
    :param last: dataframe consisting of the last day only
    :return: all decided using the abs of the spikes to compare against the threshold, corresponding to the above dfs
    """
    total_spikes = total[abs(total.diffs)>THRES]
    last_two_spikes = last_two[abs(last_two.diffs)>THRES]
    second_last_spikes = second_last[abs(second_last.diffs)>THRES]
    last_spikes = last[abs(last.diffs)>THRES]
    second_last_spikes = second_last_spikes[['date','year_month','diffs']]
    last_spikes = last_spikes[['date','year_month', 'diffs']]
    both_day_spikes = pd.merge(second_last_spikes, last_spikes, left_on='year_month', right_on='year_month')
    return total_spikes, last_two_spikes, second_last_spikes, last_spikes, both_day_spikes


def sign_spikes(second_last, last):
    """
    :param second_last: dataframe consisting of the second-last day
    :param last: dataframe consisting of the last day only
    :return: four dataframes considering the signs of spikes in each dataframe
    """
    sec_last_minus, sec_last_plus, last_minus, last_plus = sign_spikes_pre(second_last, last)
    """using outer merge"""
    """--"""
    last_two_1 = pd.merge(sec_last_minus, last_minus, left_on='year_month', right_on='year_month')
    """-+"""
    last_two_2 = pd.merge(sec_last_minus, last_plus, left_on='year_month', right_on='year_month')
    """+-"""
    last_two_3 = pd.merge(sec_last_plus, last_minus, left_on='year_month', right_on='year_month')
    """++"""
    last_two_4 = pd.merge(sec_last_plus, last_plus, left_on='year_month', right_on='year_month')

    return last_two_1, last_two_2, last_two_3, last_two_4


def sign_spikes_pre(second_last, last):
    """
    :param second_last: dataframe consisting of the second-last day
    :param last: dataframe consisting of the last day only
    :return: the dataframes consisting of spikes only, to work with in further steps
    """
    sec_last_minus = second_last[second_last.diffs < (-THRES1)]
    sec_last_minus = keep_2_col(sec_last_minus)
    sec_last_plus = second_last[second_last.diffs > THRES1]
    sec_last_plus = keep_2_col(sec_last_plus)
    last_minus = last[last.diffs <= (-THRES2)]
    last_minus = keep_2_col(last_minus)
    last_minus = last_minus.rename(columns={'diffs': 'diffs_last'})
    last_plus = last[last.diffs > THRES2]
    last_plus = keep_2_col(last_plus)
    last_plus = last_plus.rename(columns={'diffs': 'diffs_last'})
    return sec_last_minus, sec_last_plus, last_minus, last_plus


def last_day_processing(last_day):
    """
    :param last_day: dataframe consisting of last day only
    :return: dataframes with last day diff being + and -, to work with in further steps
    """
    last_day_plus = last_day[last_day.diffs > 0]
    last_day_plus = last_day_plus.rename(columns={'diffs': 'diffs_last'})
    last_day_minus = last_day[last_day.diffs <= 0]
    last_day_minus = last_day_minus.rename(columns={'diffs': 'diffs_last'})
    return last_day_plus, last_day_minus


def sign_spikes_updown(second_last, last):
    """
    :param second_last: dataframe consisting of the second-last day
    :param last: dataframe consisting of the last day only
    :return: 4 dataframes as permutations of situations given second-last-day spike
    """
    last_plus, last_minus = last_day_processing(last)
    sec_last_minus, sec_last_plus, _, _ = sign_spikes_pre(second_last, last)
    """--"""
    last_two_1 = pd.merge(sec_last_minus, last_minus, left_on='year_month', right_on='year_month')
    """-+"""
    last_two_2 = pd.merge(sec_last_minus, last_plus, left_on='year_month', right_on='year_month')
    """+-"""
    last_two_3 = pd.merge(sec_last_plus, last_minus, left_on='year_month', right_on='year_month')
    """++"""
    last_two_4 = pd.merge(sec_last_plus, last_plus, left_on='year_month', right_on='year_month')

    return last_two_1, last_two_2, last_two_3, last_two_4


def keep_2_col(df):
    """
    :param df: dataframe
    :return: to modify the dataframe
    """
    df = df[['year_month', 'diffs']]
    return df


def mean_diff(second_last_spikes, last_day):
    """
    :param second_last_spikes: dataframe consisting of second last day with spikes on them
    :param last_day: dataframe consisting of last day
    :return: the mean diffs on last day when second-last day has seen a spike
    """
    # given the second_last_spikes where based on abs calculation
    last_day = last_day[['year_month', 'diffs']]
    last_day = last_day.rename(columns={'diffs': 'diffs_last'})
    sec_last = pd.merge(second_last_spikes, last_day, left_on='year_month', right_on='year_month')
    return sec_last['diffs_last'].mean()


def get_first_day(gg_rate):
    """
    :param gg_rate: original datafram
    :return: a dataframe containing day 1 for every month data and is in the same month with last month (to compare with
    last day in last month data)
    """
    gg_rate = gg_rate[['date', 'm12']]
    gg_m12 = gg_rate['m12']
    gg_rate['diffs'] = gg_m12.diff()
    gg_rate['month_start_date'] = pd.Series([x - relativedelta(months=+1) for x in gg_rate['date']])
    gg_rate['year_month'] = gg_rate['month_start_date'].apply(lambda x: 100 * x.year + x.month)
    month_groups = gg_rate.groupby('year_month')
    first_day = month_groups.head(1)
    first_day = first_day[['diffs','year_month']]
    first_day = first_day[1:]
    return first_day


def end_start(last_day_spikes, first_day):
    last_day_spikes = last_day_spikes.iloc[:-1]
    first_day = first_day.rename(columns={'diffs':'next_month'})
    last_first = pd.merge(last_day_spikes, first_day, left_on = 'year_month', right_on='year_month')
    total = len(last_first)
    opp = 0
    for i in range(total):
        if last_first['diffs'][i]*last_first['next_month'][i]<0:
            opp += 1
    return opp/total


def get_first_day_perform(second_last, first_day):
    com1, com2, com3, com4 = sign_spikes_updown(second_last, first_day)
    # print(com1)
    # print(com2)
    # print(com3)
    # print(com4)
    dd = len(com1)/(len(com1)+len(com2))
    du = len(com2)/(len(com1)+len(com2))
    ud = len(com3)/(len(com3)+len(com4))
    uu = len(com4)/(len(com3)+len(com4))
    pos = pd.concat([com3, com4])
    neg = pd.concat([com1, com2])
    pos_mean = pos.diffs_last.mean()
    neg_mean = neg.diffs_last.mean()

    return dd, du, ud, uu, pos_mean, neg_mean


def denoise(y_data, n=5):
    b = [1.0/n] * n
    a = 1
    return filtfilt(b,a,y_data)


def find_extrema(x):
    return argrelextrema(x.values, np.greater), argrelextrema(x.values, np.less)


def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 1.5: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions


def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - .03, 1.02*t, '%d'%int(y),rotation=0, color='blue', fontsize=13)
        if y != t:
            axis.arrow(x, t+20,0,y-t, color='blue',alpha=0.2, width=txt_width*0.0,
                       head_width=.02, head_length=txt_height*0.5,
                       zorder=0,length_includes_head=True)

