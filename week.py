import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from utils import *


def get_week(df):
    """
    to get the dataframe grouped by weeks
    :param df: the original dataframe
    :return: a dataframe
    """
    df = df[['date', 'm12']]

    df['diffs'] = df['m12'].diff()

    df['year_week'] = df['date'].apply(lambda x: x.year*100 +x.isocalendar()[1])

    # df['weekday'] = df['date'].apply(lambda x: x.weekday())
    # week_groups = df.groupby('year_week')
    df = df[['date', 'year_week',  'diffs']]
    return df


def week_stat(df):
    """
    to count the up/down diffs in each week
    :param df: the week_group dataframe obtained from previous steps
    :return: a dataframe with statistic figures
    """
    weeks = sorted(set(df.year_week.values))
    pos_counts = []
    neg_counts = []
    for week in weeks:
        p = 0
        n = 0
        ds = df[df['year_week'] == week]
        ds = ds['diffs']
        for d in ds:
            if d>0:
                p += 1
            else:
                n += 1
        pos_counts.append(p)
        neg_counts.append(n)

    data = pd.DataFrame({'neg_counts': neg_counts, 'pos_counts': pos_counts, 'week':weeks})
    # data.to_excel('counts.xlsx')
    return data

    # df = df.groupby(['year_week'])
    # print(df.head())
    # # df = df.diffs.apply(lambda x: pd.Series([(x < 0).sum(), (x >= 0).sum()])).unstack()
    # # #
    # # df.to_excel('stats.xlsx')
    # # print(df)
    #
    # return


def get_days_group(df):
    """
    to get a dataframe grouped by the starting day of each group, and the neg/pos counts of diffs in each group
    :param df: the processed df of original gg_rate data
    :return: a dataframe contains grouping information
    """
    days_group = pd.DataFrame()
    days_group['group_ending_day'] = df['date']
    days_group_pos = []
    days_group_neg = []
    for i in range(len(df['date'])):
        if i < GROUP_DAYS-1:
            days_group_pos.append(None)
            days_group_neg.append(None)
            continue
        pos = 0
        neg = 0
        for j in range(GROUP_DAYS):
            d = df.iloc[i-j]['diffs'].item()
            if d > 0:
                pos += 1
            else:
                neg += 1
        days_group_pos.append(pos)
        days_group_neg.append(neg)

    days_group['pos'] = pd.Series(days_group_pos)
    days_group['neg'] = pd.Series(days_group_neg)

    return days_group


def tend_stat(df, exdf,  n=4):
    """
    to return the distribution of up/down turnover around each peak/valley
    :param df: tend_groups (got from previous steps)
    :param exdf: extrema dataframe
    :param n: number of days considered to be around a peak/valley value
    :return: another df containing info about the up/down turnover stats around the interested points
    """
    turn_over = []
    direct = []
    flag = df['tend'][0]
    for i in range(len(df)):
        if flag != df.iloc[i]['tend']:
            turn_over.append(df.iloc[i]['group_ending_day'])
            if flag == 'down':
                direct.append(-1)
            else:
                direct.append(1)
            flag = df.iloc[i]['tend']
    turnover = pd.DataFrame({'date': turn_over, 'turnover': direct})
    ex_turn = pd.merge(turnover, exdf.rename(columns={'extrema': 'predict'}),
                       left_on='date', right_on='date', how='outer')
    ex_turn = ex_turn.sort_values(by='date').reset_index(drop=True)

    return ex_turn






