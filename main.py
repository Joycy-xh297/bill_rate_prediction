import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from week import *
from utils import *
from visulisation import *
from dateutil.relativedelta import *


PATH = "RateSource2/"
SUFFIX = "" #xls for fe, xlsx for gg
FILE = "Rate_GG.xlsx"
FILE2 = "fe_price.xls"
FILE3 = "GuokaiBank_rate.xls"
THRES = 0.05
THRES1 = 0.05 #for second last day
THRES2 = 0.05 #for last day
GROUP_DAYS = 5


def gg_price():

    print("---------------------parameters---------------------")
    print("thres for abs: {}".format(THRES))
    print("thres for signed second-last day: {}".format(THRES1))
    print("thres for signed last day: {}\n".format(THRES2))

    gg_rate = get_df(PATH, FILE)
    week_group = get_week(gg_rate)

    # week_group.to_excel('wg.xlsx')
    #
    # return
    # week_group = week_group.pivot(columns='',
    #                               index='',
    #                               values='')
    # week_groups = week_group
    # print(week_groups.unique())
    # print(week_group[week_group['year_week']==201902])
    # return

    print("---------------------weeks stats---------------------")
    week_count = week_stat(week_group)
    up_week = week_count[week_count['pos_counts'] >= week_count['neg_counts']]
    down_week = week_count[week_count['pos_counts'] < week_count['neg_counts']]
    print('when week going up, the avg no of days up is {}'.format(up_week['pos_counts'].mean()))
    print('when week going down, the avg no of days down is {}\n'.format(down_week['neg_counts'].mean()))

    first_day = get_first_day(gg_rate)
    gg_rate, month_groups, last_two_days, second_last, last_day, first_day_1 = work_gg(gg_rate)

    """filtering out the spikes in abs"""

    gg_spikes, gg_month_end_spikes, second_last_spikes, last_day_spikes, both_day_spikes \
        = abs_spikes(gg_rate, last_two_days, second_last, last_day)

    first_day_spikes = first_day_1[first_day_1['diffs']>THRES]

    spikes_combine = pd.concat([second_last_spikes, last_day_spikes, first_day_spikes])
    spikes_combine = spikes_combine[['date', 'diffs']]
    """
        print("---------------------inter_month stats---------------------")
        prob = end_start(second_last_spikes, first_day)
        print("the probability of turning over next month given a spike at the end of last month is : {}".format(prob))

        dd, du, ud, uu, pm, nm= get_first_day_perform(second_last, first_day)
        print("when neg spike, prob of next month down is {}".format(dd))
        print("when neg spike, prob of next month up is {}".format(du))
        print("mean diffs: {}".format(nm))
        print("when pos spike, prob of next month down is {}".format(ud))
        print("when pos spike, prob of next month up is {}".format(uu))
        print("mean diffs: {}\n".format(pm))

        # statistics for abs
        total_spikes = len(gg_spikes)
        sec_abs_spikes = len(second_last_spikes)
        spikes_distribution = len(gg_month_end_spikes)/total_spikes
        probability = len(both_day_spikes)/sec_abs_spikes

        print("---------------------month_end statistics------------------------")
        print("statistics with abs")
        print("given a spike on the second-last day of a month, "
              "the probability of having another spike on the last day of month is: {}".format(probability))
        print("the proportion of distribution of spikes on the month ends: {}\n".format(spikes_distribution))

        # plotting
        # plt.bar(gg_spikes.year_month, abs(gg_spikes.diffs))
        # plt.plot(gg_rate.year_month,[THRES]*len(gg_rate), color = 'r')
        # plt.show()


        both_1, both_2, both_3, both_4 = sign_spikes(second_last_spikes, last_day)
        both_5, both_6, both_7, both_8 = sign_spikes_updown(second_last_spikes, last_day)
        mean1 = both_5.diffs_last.mean()
        mean2 = both_6.diffs_last.mean()
        mean3 = both_7.diffs_last.mean()
        mean4 = both_8.diffs_last.mean()
        sec_last_minus, sec_last_plus, _, _ = sign_spikes_pre(second_last_spikes, last_day)


        prob1 = len(both_1)/len(sec_last_minus)
        prob2 = len(both_2)/len(sec_last_minus)
        prob3 = len(both_3)/len(sec_last_plus)
        prob4 = len(both_4)/len(sec_last_plus)
        prob5 = len(both_5)/len(sec_last_minus)
        prob6 = len(both_6)/len(sec_last_minus)
        prob7 = len(both_7)/len(sec_last_plus)
        prob8 = len(both_8)/len(sec_last_plus)
        mean_last_day1 = mean_diff(sec_last_minus, last_day)
        mean_last_day2 = mean_diff(sec_last_plus, last_day)

        print("when second_last_spike appears to be -, the probability of last day spike being - is: {}".format(prob1))

        print("when second_last_spike appears to be -, the probability of last day spike being + is: {}".format(prob2))
        print("given a negative spike on the second-last day of a month, the mean diffs on last day of month is: {}".format(
            mean_last_day1))
        print("the probability of negative last day diffs is {}, and the mean diffs is: {}".format(prob5,mean1))
        print("the probability of positive last day diffs is {}, and the mean diffs is: {}\n".format(prob6,mean2))

        print("when second_last_spike appears to be +, the probability of last day spike being - is: {}".format(prob3))
        print("when second_last_spike appears to be +, the probability of last day spike being + is: {}".format(prob4))
        print("given a positive spike on the second-last day of a month, the mean diffs on last day of month is: {}".format(
            mean_last_day2))
        print("the probability of negative last day diffs is {}, and the mean diffs is: {}".format(prob7, mean3))
        print("the probability of positive last day diffs is {}, and the mean diffs is: {}\n".format(prob8, mean4))

    """
    return spikes_combine


def tend():
    gg_rate = get_df(PATH, FILE)
    gg_rate = gg_rate[['date','m12']]
    gg_rate = gg_rate.rename(columns={'m12': 'rate'})

    # if want all original data to produce a fit, comment the following lines:
    spikes_combined = gg_price()
    indices = list(spikes_combined.index)
    gg_rate.drop(gg_rate.index[indices], inplace=True)

    # fitting
    gg_rate['smoothed'] = denoise(gg_rate['rate'])

    max_inds, min_inds = find_extrema(gg_rate['smoothed'])
    maxs = gg_rate.iloc[max_inds]
    max_dates = maxs['date']
    maximas = maxs['smoothed']
    mins = gg_rate.iloc[min_inds]
    min_dates = mins['date']
    minimas = mins['smoothed']

    max_df = pd.DataFrame({'date': max_dates, 'extrema': [1] * len(max_dates)})
    min_df = pd.DataFrame({'date': min_dates, 'extrema': [-1]* len(min_dates)})
    extrema = pd.concat([max_df, min_df])
    gg_extrema = pd.merge(gg_rate, extrema, left_on='date', right_on='date', how='outer')
    # print(gg_extrema.head())
    # gg_extrema.to_excel('gg_extrema_8.xlsx')

    """

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.plot(max_dates, maximas, 'o', color='r')
    ax.plot(min_dates, minimas, 'd', color='b')
    ax.grid()


    # txt_height = 0.04 * (plt.ylim()[1] - plt.ylim()[0])
    # txt_width = 0.02 * (plt.xlim()[1] - plt.xlim()[0])
    # text_positions = get_text_positions(dates, vals, txt_width, txt_height)
    # text_plotter(dates, vals, text_positions, ax, txt_width, txt_height)

    plt.plot(gg_rate.date, gg_rate.smoothed, color='r')
    plt.plot(gg_rate.date, gg_rate.rate, color='b', alpha=0.4)
    plt.savefig('smoothed_extrema_8.png')

    plt.show()
    # print(gg_rate.head())
    """

    return gg_rate, extrema


def fe_price():
    fe = get_df(PATH, FILE2)
    fe = fe[['date', 'close_price']]
    fe = fe.dropna()
    gg = get_df(PATH, FILE)
    gg = gg[['date', 'm12']]
    gg = gg.rename(columns={'m12': 'gg_rate'})
    fe_gg = pd.merge(fe, gg, left_on='date', right_on='date')
    fe_gg.to_excel('compare.xlsx')

    # plt.plot(fe_gg.date, fe_gg.close_price, color='b')
    # plt.plot(fe_gg.date, fe_gg.m12*1000, label='gg_rate*1000', color='r')
    # plt.legend()
    # plt.savefig('compare.png')
    # plt.show()
    """get the corr factor of the prices and rates"""
    corr1 = fe_gg.corr()
    print("-------------------corr of the prices-----------------------")
    print(corr1)

    """get the corr factor of the change rates"""
    fe_gg['diffs_close'] = fe_gg['close_price'].pct_change()
    fe_gg['diffs_gg'] = fe_gg['gg_rate'].pct_change()
    corr2 = fe_gg[['diffs_close', 'diffs_gg']].corr()

    plt.plot(fe_gg.date, fe_gg.diffs_close, color='b')
    plt.plot(fe_gg.date, fe_gg.diffs_gg, label='gg_rate*1000', color='r')
    plt.legend()
    # plt.savefig('compare_diffs.png')
    plt.show()

    print("-------------------corr of the diffs-----------------------")
    print(corr2)


def gk_rate():
    gg_rate = get_df(PATH, FILE)
    gg_rate = gg_rate[['date', 'm12']]
    gk_rate = get_df(PATH, FILE3)
    gk_rate = gk_rate[['date', 'Debt_1Y']]
    gk_gg = pd.merge(gk_rate, gg_rate, left_on='date', right_on='date')
    gk_gg['gk_diffs'] = gk_gg['Debt_1Y'].diff()
    gk_gg['gg_diffs'] = gk_gg['m12'].diff()
    gk_gg_diff = gk_gg[['gk_diffs', 'gg_diffs']]
    gk_gg = gk_gg[['date', 'Debt_1Y', 'm12']]
    plt.plot(gk_gg.date, gk_gg.m12, color='r')
    plt.plot(gk_gg.date, gk_gg.Debt_1Y, color='b')
    plt.show()
    print(gk_gg.corr())
    print(gk_gg_diff.corr())


def tend_days():
    gg_rate_fit, extrema = tend()
    extrema = extrema.sort_values(by='date')
    extrema = extrema.reset_index()

    gg_rate = get_df(PATH, FILE)
    gg_rate['diffs'] = gg_rate.m12.diff()
    gg_rate = gg_rate[['date', 'diffs', 'm12']]
    gg_rate = gg_rate.rename(columns={'m12': 'rate'})

    days_group = get_days_group(gg_rate)
    days_group = days_group[['group_ending_day', 'pos', 'neg']]
    days_group['tend'] = days_group.apply(lambda x: 'up' if (x.pos > x.neg) else 'down', axis=1)

    inds_to_drop = []
    for i in range(len(extrema)-1):
        date1 = extrema.iloc[i].date
        date2 = extrema.iloc[i+1].date
        # ind2 = gg_rate[gg_rate['date'] == date2].index
        # ind1 = gg_rate[gg_rate['date'] == date1].index
        ind1 = extrema['index'][i]
        ind2 = extrema['index'][i+1]
        if (ind2 - ind1) < GROUP_DAYS:
            inds_to_drop.append(i)
            inds_to_drop.append(i+1)
    extrema.drop(extrema.index[inds_to_drop], inplace=True)

    # days_group['group_ending_day'] = pd.concat([days_group['group_ending_day'],
    #                                             days_group['group_ending_day']]).drop_duplicates()

    tend_groups = pd.merge(days_group, extrema, left_on='group_ending_day',
                           right_on='date', how='outer')
    tend_groups = tend_groups.sort_values(['group_ending_day', 'date'])
    tend_groups = tend_groups[(GROUP_DAYS-1):].reset_index(drop=True)

    # tend_groups.to_excel('tend_groups.xlsx')
    ex_turn1 = tend_stat(tend_groups, extrema)
    ex_turn1 = ex_turn1[['date', 'turnover', 'predict']]
    # print(ex_turn)
    # print(gg_rate.tail())
    ex_turn1['ind'] = ex_turn1['date'].apply(lambda x: gg_rate[gg_rate['date'] == x].index.item())

    # ex_turn.to_excel('ex_turn_less_fit.xlsx')
    # ex_turn.to_excel('without_endstart_spikes.xlsx')

    # using 'test' instead of the fitted extremas:
    test = pd.read_excel('test.xlsx')
    test = test[test['peak'] != 0][['date', 'peak']].rename(columns={'peak': 'extrema'})
    ex_turn = tend_stat(tend_groups, test)
    ex_turn['ind'] = ex_turn['date'].apply(lambda x: gg_rate[gg_rate['date'] == x].index.item())
    # ex_turn.to_excel('ex_turn_test.xlsx')

    tend_plot(gg_rate, gg_rate_fit, ex_turn1, ex_turn)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # gg_price()
    # fe_price()
    # tend()
    # gk_rate()
    tend_days()
    # gg_rate = get_df(PATH, FILE)
    # gg_rate = gg_rate[['date', 'm12']]
    # gg_rate.to_csv('gg_rate.csv')






