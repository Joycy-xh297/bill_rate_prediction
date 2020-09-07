import pandas as pd
import datetime
import os
def main():
    RateGG=pd.read_excel(os.path.join('../Bills_Processing/RateSource2','Rate_GG.xlsx'))
    RateGG['date']=RateGG.date.map(lambda dt:dt.date()); # date,
    RateGG=RateGG[['date','m12']];
    RateGG['month']=RateGG.date.map(lambda dt:dt.strftime("%Y%m"));

    tradeMonth=sorted(set(RateGG.month.values));
    Data=pd.DataFrame([])
    for mth in tradeMonth:
        tem=RateGG[RateGG.month==mth].copy()
        tem.sort_values(by=['date'],ascending=True,inplace=True);
        tem['rank_dt']=range(1,len(tem)+1);# rank_dt; 正序排序日
        tem.sort_values(by=['date'],ascending=False,inplace=True);
        tem['rank_dt2']=range(1,len(tem)+1); # rank_dt2 倒序排序日
        Data=Data.append(tem)
    Data.sort_values(by=['date'],ascending=True,inplace=True);

    tradeDays=sorted(set(Data.date.values))
    tem=Data[['date','m12']].copy();tem.columns=['last_dt','last_m12']
    Data['last_dt']=Data.date.map(lambda x: tradeDays[tradeDays.index(x)-1] if tradeDays.index(x)>0 else (x-datetime.timedelta(days=1)))
    Data=pd.merge(Data,tem,on=['last_dt'],how='inner');
    Data['delta']=Data['m12']-Data['last_m12']
    print(Data)
    ## 倒数第二天跳空时，倒数第一天的变化；
    jump=0.06;
    Data['jump']=Data.delta.map(lambda x: tag_jump(x,jump));
    # 分2 类，分别处理跳涨和跳跌
    jumpDaysUP2=Data[(Data.rank_dt2==2) &(Data.jump==1)].date.values;
    jumpDaysDown2 = Data[(Data.rank_dt2 == 2) & (Data.jump == -1)].date.values;
    JumpUp2=pd.DataFrame([]);JumpDown2=pd.DataFrame([])
    for dt in jumpDaysUP2:
        tem=Data[Data.date==tradeDays[tradeDays.index(dt)+1]].copy();
        JumpUp2=JumpUp2.append(tem)
    for dt in jumpDaysDown2:
        tem = Data[Data.date == tradeDays[tradeDays.index(dt) + 1]].copy();
        JumpDown2=JumpDown2.append(tem)
    NumUP2=len(JumpUp2);
    UPUP=len(JumpUp2[JumpUp2.delta>0]);UPDown=len(JumpUp2[JumpUp2.delta<0]);avg1_UP2=JumpUp2.delta.mean();
    NumDown2=len(JumpDown2);
    DownUP=len(JumpDown2[JumpDown2.delta>0]);DownDown=len(JumpDown2[JumpDown2.delta<0]);avg1_Down2=JumpDown2.delta.mean();
    print('=' * 20)
    print ('for -2 day jump up, total num is '+str(NumUP2)+': on -1 day, up rate='+str(UPUP/float(NumUP2))+ ' down rate ='+str(UPDown/float(NumUP2)))
    print ('avg fluctuation on -1 is ' +str(avg1_UP2))
    print ('='*20)
    print ('for -2 day jump down, total num is '+str(NumDown2)+': on -1 day, up rate='+str(DownUP/float(NumDown2))+ ' down rate ='+str(DownDown/float(NumDown2)))
    print ('avg fluctuation on -1 is ' +str(avg1_Down2))

    # ## 月末后2天跳空后，月初反弹的概率（后两天涨跌概率之和）
    jump2=0.06;
    tem=Data[['date','delta']].copy();tem.columns=['last_dt','lag_delta']
    Data=pd.merge(Data,tem,on=['last_dt'],how='inner');
    Data['delta_sum']=Data.delta+Data.lag_delta;
    Data['consistent']=[ 1 if x*y>0 else 0 for (x,y) in zip(Data.delta,Data.lag_delta)];
    Data['jump2']=Data.delta_sum.map(lambda x:tag_jump(x,jump2))
    DaysUP=Data[(Data.rank_dt2==1) & (Data.consistent==1) &(Data.jump2==1)].date.values
    DaysDown=Data[(Data.rank_dt2==1) & (Data.consistent==1) & (Data.jump2==-1)].date.values
    UPData=pd.DataFrame([]);DownData=pd.DataFrame([])
    for dt in DaysUP:
        tem=Data[Data.date==tradeDays[tradeDays.index(dt)+1]];
        UPData=UPData.append(tem)
    for dt in DaysDown:
        tem=Data[Data.date==tradeDays[tradeDays.index(dt)+1]];
        DownData=DownData.append(tem);

    NumUP=len(UPData);UPUP=len(UPData[UPData.delta>0]);UPDown=len(UPData[UPData.delta<0]);
    avg_UP=UPData.delta.mean();
    print('=' * 20)
    print('for consistent jumping down in last 2 days of one month, total num is ' + str(
        NumUP) + ' and for the next day upRate= ' + str(float(UPUP) / NumUP)
          + ' downRate= ' + str(float(UPDown) / NumUP) )
    print('  avg fluctuation on T1 is ' + str(avg_UP));

    NumDown=len(DownData);DownUp=len(DownData[DownData.delta>0]);DownDown=len(DownData[DownData.delta<0]);
    avg_Down=DownData.delta.mean();
    print('=' * 20)
    print ( 'for consistent jumping down in last 2 days of one month, total num is '+str(NumDown)+' and for the next day upRate= '+str(float(DownUp)/NumDown)
            +' downRate= '+str(float(DownDown)/NumDown))
    print(' avg fluctuation on T1 is ' +str(avg_Down));




def tag_jump(x,jump):
    if x>jump:
        return (1)
    elif x<-jump:
        return (-1);
    else:
        return (0)


if __name__=='__main__':

    main()