import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from main import *
from week import *
from utils import *


def tend_plot(df, fit, df1, df2):
    """
    to plot the results of the functino tend_days
    :param df: the original dataframe gg_rate
    :param fit: the original dataframe gg_rate with the already fitted column "smoothed"
    :param df1: the turnover - prediction contrast dataframe1 (ex_turn1) using the fitted extremas
    :param df2: the turnover - prediction contrast dataframe1 (ex_turn) using the observed extremas from real data
    :return: none
    """
    df1['rate'] = df1['ind'].apply(lambda x: df.iloc[x].rate)
    df2['rate'] = df2['ind'].apply(lambda x: df.iloc[x].rate)

    plt.figure(figsize=(60, 40))

    plt.subplot(2, 1, 1)
    plt.title('predict using the fitted line with end_start_spikes removed', fontsize=20)
    plt.plot(df.date, df.rate, color='b', alpha=0.4)  # plotting the original data
    plt.plot(fit.date, fit.smoothed, color='g', alpha=0.4, linewidth=4)  # plotting the fitted line
    plt.plot(df1.loc[(df1['predict'] == 1) | (df1['predict'] == -1)].date,
             df1.loc[(df1['predict'] == 1) | (df1['predict'] == -1)].rate,
             'o', color='r', markersize=10, label='predict')
    for time in df1.loc[(df1['predict'] == 1 ) | ( df1['predict'] == -1)].date:
        plt.axvline(time, color='r', linewidth=1, alpha=0.5)
    plt.plot(df1.loc[(df1['turnover'] == 1) | (df1['turnover'] == -1)].date,
             df1.loc[(df1['turnover'] == 1) | (df1['turnover'] == -1)].rate,
             'd', color='b', markersize=7, label='real_data')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('predict using the test data', fontsize=20)
    plt.plot(df.date, df.rate, color='b', alpha=0.4)  # plotting the original data
    plt.plot(fit.date, fit.smoothed, color='g', alpha=0.4, linewidth=4)  # plotting the fitted line
    plt.plot(df2.loc[(df2['predict'] == 1) | (df2['predict'] == -1)].date,
             df2.loc[(df2['predict'] == 1) | (df2['predict'] == -1)].rate,
             'o', color='r', markersize=10, label='predict')
    for time in df2.loc[(df2['predict'] == 1) | (df2['predict'] == -1)].date:
        plt.axvline(time, color='r', linewidth=1, alpha=0.5)
    plt.plot(df2.loc[(df2['turnover'] == 1) | (df2['turnover'] == -1)].date,
             df2.loc[(df2['turnover'] == 1) | (df2['turnover'] == -1)].rate,
             'd', color='b', markersize=7, label='real_data')

    plt.legend()
    plt.savefig('compare_ex_turn.png')
    plt.show()