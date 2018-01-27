# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import sys,os
reload(sys)
sys.setdefaultencoding('utf-8')
import pdb
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import   RandomForestClassifier
#from sklearn.cross_validation import train_test_split
#from sklearn.metrics import confusion_matrix
#from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
#import sklearn
import datetime
#import collections
#import graphviz
#import pydot
#import StringIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from main import getDfFromDir,getTradDays
import scipy.stats.mstats as mstats
#import scipy
#import inspect, itertools
from joblib import Parallel, delayed
from collections import OrderedDict,defaultdict
import json
import matplotlib.gridspec as gridspec
from functools import partial
from tools import timeit

DIR_GEN = '/home/wyy/Alpha/STOCK_DT'
trade_days_path = DIR_GEN+'/tradedays.csv'
start_day = '20040101'
end_day = '20171208'
NJOBS = 32

PathBackTestResult = DIR_GEN + '/figure/pnl/{1}-{0}.png'
global params


Name = '20DM_price'
signal_karg = { 'pct_chg':False
                ,'window_back':7
                ,'window_back_min_num':7
                ,'rank_axis':1                                                # 默认为1
                ,'is_neutral':False
                ,'is_industry_neutral':False
                ,'expanding_rank':False
                }

exec_code = '''
VDR = a[-5:].sum()
vdr_diff1 = sum(np.diff(a[-6:],1))
sign1 = np.sign(VDR)
sign2 = np.sign(vdr_diff1)

if abs(VDR )< 0.01:
    sign1 = 0
if abs(vdr_diff1 )< 0.01:
    sign2 = 0

if sign1 == 1 and sign2 == 1:
    sign = -3 # recomd -1;不能1.收益大量降低,ir 0.13
if sign1 == 1 and sign2 == -1:
    sign = 1 # 1 rise ir,ret,tv , can cancel
if sign1 == -1 and sign2 == 1:
    sign = 1 # recomd 1, -1 down
if sign1 == -1 and sign2 == -1:
    sign = 3 # recomd 1 ;不能-1.收益大量降低
if sign1 == 0 and sign2 == -1:
    sign = 2 # 1 rise ; -1 down
if sign1 == 0 and sign2 == 1:
    sign = 2 # 1 rise ;-1 down
if sign1 == 1 and sign2 == 0:
    sign = -1 # recomd -1;1 rise ir; -1 down tv, little ir and  rise ret
if sign1 == -1 and sign2 == 0:
    sign = 1 # 1 rise ;-1 down
if sign1 == 0 and sign2 == 0:
    sign = 1 # recomd 1,keep

sig = sign * abs(VDR )

'''

exec_code = '''

#sig = -sum(np.diff(a[-7:],2))
#sig = -sum(np.diff(a[-6:],1) > -0.0017)
#sig = -sum(np.diff(a[-5:],1))
#sig = -(a[-1] - a[0])
sig = -(np.diff(a,2).sum())
'''

exec_code = '''
sig = sum(mstats.rankdata(a)[-15:])
'''

def get_signal(a):
    a = a[:-1]

#    IV_dr = -np.sum(a)
#    IV_dr_price = -(a[-1] - a[0])
#    price = a[-1]
#    diff1 = -sum(np.diff(a,1))
#    diff1_abs = abs(-sum(np.diff(a,1)))
#    diff2 = -sum(np.diff(a,2))
#    diff3 = -sum(np.diff(a,3))
#    XVIII_dr = np.sum(a)
#    ptp = np.max(a) - np.min(a)
#    ptp = abs(np.max(a)) + abs(np.min(a))

# =============================================================================
#     vdr = sum(a[:5])
#     vdr_diff1 = sum(np.diff(a[:6],1))
#
#     sign1 = np.sign(vdr)
#     sign2 = np.sign(vdr_diff1)
#
#     if sign1 == 1 and sign2 == 1:
#         sign =  -1
#     if sign1 == -1 and sign2 == -1:
#         sign =  1
#     if sign1 == 1 and sign2 == -1:
#         sign =  -1
#     if sign1 == -1 and sign2 == 1:
#         sign =  1
#     if sign2 == 0:
#         sign = -sign1
#     if sign1 == 0:
#         sign = -sign2
#
#     sig = sign*( abs(vdr) )
# =============================================================================
# =============================================================================
# XXX = a[:30]
# BXX = a[:120]
# sig = XXX.mean() - BXX.mean()
# =============================================================================
#    baoyi = IV_dr+diff1
#    baoyi = IV_dr+diff1*1.8
#sig = vdr/(abs(vdr_diff1)+abs(vdr))
#sig = np.power(vdr_diff1,0.5)

    exec(exec_code)

    return sig

def my_decorator(func):
    def wrapper(*args, **kwargs):
#        args_name = list(OrderedDict.fromkeys(inspect.getargspec(func)[0] + kwargs.keys()))
#        args_dict = OrderedDict(list(itertools.izip(args_name, args)) + list(kwargs.items()))
        args_dict = OrderedDict(list(kwargs.items()))
        global params
        params = args_dict
#        print args_dict
        return func(*args, **kwargs)
    return wrapper

def func_expand(sr):
    sr = sr.expanding().apply(lambda x:mstats.rankdata(x)[-1]/len(x))
    return sr

def func_rolling(window_back,window_back_min_num,sr):
    sr = sr.rolling(window = window_back+1
                    ,min_periods = window_back_min_num + 1).apply(lambda x:get_signal(x))
    return sr



#@timeit
@my_decorator
def signal(df,pct_chg=True,window_back=5,window_back_min_num=5,rank_axis=1,expanding_rank=False,is_neutral=True,is_industry_neutral=True):

    if pct_chg == True:
        df = df.pct_change()
    elif pct_chg == False:
        df = df
    else:
        raise AssertionError('please input valid paramter pct_chg in [0,1]')

    if not rank_axis in [0,1]:
        raise AssertionError('please input valid paramter rank_axis in [0,1]')

#    df_signal  = df.rolling(window = window_back+1
#                            ,min_periods = window_back_min_num + 1).apply(lambda x:get_signal(x))

    col_name = df.columns
    func_ = partial(func_rolling,window_back,window_back_min_num)
    sr_list = Parallel(n_jobs=NJOBS, verbose=2)( delayed(func_)(col[1]) for col in df.iteritems())
    df_signal = pd.concat(sr_list,axis=1)
    df_signal = df_signal[col_name]


    if is_neutral == True:
        if expanding_rank == False:
            df_signal_rank = df_signal.rank(method = 'average'                                      # 相同值取排位均值
                                            ,na_option ='keep'                                      # 對nan保留
                                            ,pct=True                                               # 計算百分比，從0-1
                                            ,axis = rank_axis
                                            )
        elif expanding_rank == True:
    #        pdb.set_trace()
            col_name = df_signal.columns
            sr_list = Parallel(n_jobs=NJOBS, verbose=2)( delayed(func_expand)(col[1]) for col in df_signal.items())
            df_signal_rank = pd.concat(sr_list,axis=1)
            df_signal_rank = df_signal_rank[col_name]
        else:
            raise AssertionError('please input valid paramter expanding_rank in [False,True]')
    else:
         return df_signal

    df_signal_rank_map = 2*(df_signal_rank - 0.5)
#    pdb.set_trace()
    df_signal_rank_map = df_signal_rank_map.applymap(lambda x : pow(abs(x),1.5)*np.sign(x) if not np.isnan(x) else x )
#    df_signal_rank_map = df_signal_rank_map.pow(1.5)
#    df_signal_rank_map.loc['2015-04-21':'2015-06-01',:].iloc[:,:4]

    if is_industry_neutral == True:
        df_signal_rank_neutral = neutral(df_signal_rank_map)
    elif is_industry_neutral == False:
        df_signal_rank_neutral = df_signal_rank_map
    else:
        raise AssertionError('please input valid paramter is_industry_neutral in [True,False]')

    df_signal_rank_neutral = df_signal_rank_neutral.applymap(lambda x : pow(abs(x),1.5)*np.sign(x) if not np.isnan(x) else x )
#    df_signal_rank_map = df_signal_rank_map.pow(1.5)


    return df_signal_rank_neutral


def max_drawdown(sr):
    i = np.argmax(np.maximum.accumulate(sr) - sr) # end of the period
    j = np.argmax(sr[:i]) # start of period
    return abs(sr[i] - sr[j] )

@timeit
def neutral(df_signal):
    industry_ids = pd.unique(pd.Series(df_industry.values.ravel()).dropna())
    industry_ids.sort()
    l = []
    for industry_id  in industry_ids:
        df_industry_one = df_industry == industry_id
        df_industry_one = df_industry_one.replace(False,np.nan)
#            pdb.set_trace()
        df_signal_one_indstry = df_industry_one * df_signal
        sr_signal_sum_one_indstry = df_signal_one_indstry.mean(1)
#            pdb.set_trace()
        df_signal_one_indstry_neutral = df_signal_one_indstry.sub(sr_signal_sum_one_indstry.values,axis=0)
        l.append(df_signal_one_indstry_neutral.fillna(0))
    return sum(l)

#@timeit
def industry_renf(df_signal,df_close_pct,df_industry,interval='Q'):
    signal_sum_perday = abs(df_signal).sum(axis=1)
    df_weight = df_signal_rank_map.div(signal_sum_perday.values,axis=0)
    df_close_pct_tomorrow = df_close_pct.shift(-1)
    df_return  = df_weight * df_close_pct_tomorrow
#    pdb.set_trace()
    industry_ids = pd.unique(pd.Series(df_industry.values.ravel()).dropna())
    industry_ids.sort()
    l = []
    for industry_id  in industry_ids:
        df_industry_one = df_industry == industry_id
        df_industry_one = df_industry_one.replace(False,np.nan)
#            pdb.set_trace()
        df_return_one_indstry = df_industry_one * df_return
        sr_return_sum_one_indstry = df_return_one_indstry.mean(axis=1)
        sr_return_sum_one_indstry.name = industry_id
        l.append(sr_return_sum_one_indstry)

    df_industry_return = pd.concat(l,axis=1)

    df_industry_return_q = df_industry_return.resample(interval).mean()
    df_industry_return_q_rank = df_industry_return_q.rank(axis=1,na_option='keep')
    df_industry_return_q_weight = df_industry_return_q_rank.div(df_industry_return_q_rank.sum(1).values,axis=0)
    df_industry_return_q_weight = df_industry_return_q_weight.shift(1)
    df_industry_return_d_weight = df_industry_return_q_weight.resample('D').bfill()
    df_industry_return_d_weight = df_industry_return_d_weight.loc[df_signal.index]

    l_ = []
    for industry_id  in industry_ids:
        sr_industry_return_d_weight = df_industry_return_d_weight[industry_id]
        df_industry_one = (df_industry == industry_id).replace(False,np.nan)
        df_industry_return_d_weight_onecode = df_industry_one.mul(sr_industry_return_d_weight.values,axis=0)
        l_.append(df_industry_return_d_weight_onecode.fillna(0))

    df_return_d_renf_weight = sum(l_)
#    pdb.set_trace()
    return df_return_d_renf_weight * df_signal


def print_ret_msg(df_return,df_weight,df_close_pct_tomorrow,sr_rev_1):

    df = pd.DataFrame()

    df['RET'] = df_return.sum(1).resample("A").sum()
    df['IR'] = df_return.sum(1).resample("A").mean()/df_return.sum(1).resample("A").std()
    df['MAXDD'] = df_return.sum(1).resample("A").apply(max_drawdown)
    df['Turnover'] = abs(df_weight.diff(1)).sum(axis=1).resample("A").mean()
    df['CORR_REV01'] = df_return.sum(1).corr(sr_rev_1)
#    df['WIN'] = (~((df_close_pct_tomorrow>0) ^ (df_weight.replace(0,np.nan)>0))).mean(axis=1).resample("A").mean()

    sr = df.mean()
    sr.name = 'Sumary'
    df = df.append(sr)

    print (Name)
    print (df)

    return df.loc['Sumary']

@timeit
def plot_top_return(df_return,df_price,sr_return_long,sr_return_short):

    Days = 15
    Code_Num = 10
    Window_Conpansation = int(params['window_back'] * 0.75)
    Window_Lenth = params['window_back'] + Window_Conpansation


    fig = plt.figure(figsize=(26,36))

    gs = gridspec.GridSpec(nrows=8, ncols=5, height_ratios=[3.5,2,1,1,1,1,1,1])

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_return.sum(1).cumsum(),label='All', color='b')
    ax1.plot(sr_return_long,label='Long', color='r')
    ax1.plot(sr_return_short,label='Short', color='g')
    ax1.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    ax1.legend(loc="upper left",shadow=True, fancybox=True)
    ax1.set_title(Name)

    ax3 = fig.add_subplot(gs[1, :])
#    pdb.set_trace()
    index_start_price = df_index.loc['000001.SH']['[8]ClosePrice'].iloc[0]
    ax3.plot(df_index.loc['000001.SH']['[8]ClosePrice'].apply(lambda sr: sr/index_start_price), color='y')
    ax3.set_title('000001.SH')

# =============================================================================
#算复利
#     ret_compound = (1 + df_return.sum(1)).cumprod()
#     ax3.plot(ret_compound, color='b')
# =============================================================================

#    code_byte = get_signal.__code__.co_code

    left, width = 0, 1
    bottom, height = 0, 1
#    right = left + width
    top = bottom + height
#    p = patches.Rectangle((left, bottom), width, height,fill=False, transform=ax1.transAxes, clip_on=False)
#    ax1.add_patch(p)

    ax1.text(0, 0.75*(bottom+top), msg_str, fontsize=18,horizontalalignment='left',verticalalignment='center',transform=ax1.transAxes)
    ax1.text(left, 0.5*(bottom+top), params_str, fontsize=18,horizontalalignment='left',verticalalignment='center',transform=ax1.transAxes)
    ax1.text(0.15, 1*(bottom+top), exec_code, fontsize=18,horizontalalignment='left',verticalalignment='top',transform=ax1.transAxes)
#    ax1.text((left+width)/2, 0.8*(bottom+top),'code', fontsize=18,horizontalalignment='center',verticalalignment='top',transform=ax1.transAxes)

#    ax1.fill_between(df_return.sum(1).cumsum().index,0,df_return.sum(1).cumsum(), alpha=0.3)
    ax1.grid(True, zorder=5)


#    win_dates = df_return.sum(1).sort_values(ascending=False).head(Days).index
    win_dates = df_return.sum(1).resample('A').apply(lambda x:x.sort_values(ascending=False).head(3)).sort_values(ascending=False).head(Days).reset_index(level=0,drop=True).index
    fail_dates = df_return.sum(1).resample('A').apply(lambda x :x.sort_values(ascending=False).tail(3)).sort_values(ascending=False).tail(Days).reset_index(level=0,drop=True).index
#    fail_dates = df_return.sum(1).sort_values(ascending=False).tail(Days).index
    win_dict = OrderedDict()
    fail_dict = OrderedDict()
    all_dic = OrderedDict()
    df_wins = df_return.loc[win_dates]
    df_fails = df_return.loc[fail_dates]
    for i in range(len(df_wins.index)):
        sr_win = df_wins.iloc[i,:]
        win_codes = sr_win.sort_values(ascending=False).head(Code_Num).index
        win_dict.setdefault(sr_win.name, []).append(win_codes)
    for i in range(len(df_fails.index)):
        sr_fail = df_fails.iloc[i,:]
        fail_codes = sr_fail.dropna().sort_values(ascending=False).tail(Code_Num).index
        fail_dict.setdefault(sr_fail.name, []).append(fail_codes)

    for d in (win_dict, fail_dict):# combine diction
        for key, value in d.items():
            all_dic.setdefault(key, []).extend(value)

    for i,dic in enumerate(all_dic.items()):
        date = dic[0]
        codes = dic[1][0]
        idx = df_price.index.get_loc(date)
        df_toplot = df_price.iloc[idx-Window_Lenth:idx+Window_Lenth+1,:][codes]

#        df_toplot = df_toplot.apply(lambda sr: sr/sr.iloc[0] - 1)
        df_toplot = df_toplot.apply(lambda sr: preprocessing.scale(sr.ffill().bfill()))

        df_toplot = df_toplot.reset_index(drop=True)
        ax2 = fig.add_subplot(gs[2+(i/5),i%5])
        ax2.plot(df_toplot,'o-', lw=2, alpha=0.7)
#        pdb.set_trace()
        ax2.set_title(date.strftime('%Y-%m-%d'))
        ax2.axvline( Window_Lenth,linewidth=2, color='r')
        ax2.axvline( Window_Lenth-1, linewidth=2, color='b',linestyle='dashed')
        ax2.axvline( Window_Conpansation, linewidth=2, color='y',linestyle='dashed')
        ax2.axvline( 2*Window_Lenth - Window_Conpansation, linewidth=2, color='y',linestyle='dashed')
#        ax2.vlines( df_price.iloc[idx].name, 0, 1, transform=ax2.get_xaxis_transform(),linewidth=2, color='b')
#        ax2.format_xdata = mdates.DateFormatter('%Y-%m-%d')
#        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
#        ax2.xaxis.set_major_locator(mdates.YearLocator())
        ax2.grid()
    return fig


def getBaseData(df):

    df = df.drop_duplicates(subset=['[2]TradingDay','[1]SecuCode'])
    df['[2]TradingDay'] = pd.to_datetime(df['[2]TradingDay'].astype('str').str.slice(0,8))
    #df.sort_index(inplace=True)
#    pdb.set_trace()
    df = df.set_index(['[2]TradingDay','[1]SecuCode'],verify_integrity=False)

    df_open = df['[4]OpenPrice']
    df_open = df_open.unstack(level=-1)
#    df_open = df_open.dropna(axis=0,how='all')

    df_close_bf = df['[26]ClosePrice_BackFill']
    df_close_bf = df_close_bf.unstack(level=-1)
#    df_close_bf = df_close_bf.dropna(axis=0,how='all')

    df_high = df['[5]HighPrice']
    df_high = df_high.unstack(level=-1)
#    df_high = df_high.dropna(axis=0,how='all')

    df_low = df['[6]LowPrice']
    df_low = df_low.unstack(level=-1)
#    df_low = df_low.dropna(axis=0,how='all')

    df_volume = df['[8]TurnoverVolume']
    df_volume = df_volume.unstack(level=-1)
#    df_volume = df_volume.dropna(axis=0,how='all')

    df_close = df['[7]ClosePrice']
    df_close = df_close.unstack(level=-1)
#    df_close = df_close.dropna(axis=0,how='all')

    df_TotalShares = df['[21]TotalShares']
    df_TotalShares = df_TotalShares.unstack(level=-1)
#    df_TotalShares = df_TotalShares.dropna(axis=0,how='all')


    df_TotalValue = df_close * df_TotalShares
    df_turnover = df_close * df_volume / df_TotalValue



    data = {'close':df_close
            ,'open':df_open
            ,'high':df_high
            ,'low':df_low
            ,'volume':df_volume
            ,'value':df_TotalValue
            ,'turnover':df_turnover}

    df_Panel = pd.Panel(data)
    df_Panel = df_Panel.dropna(axis=1,how='all')
    df_Panel = df_Panel.dropna(axis=2,how='all')
    return df_Panel

if __name__ == '__main__':

    global params

    df = pd.read_csv(DIR_GEN + r'/data/stock_history_daily_select.csv')
    df_index = pd.read_csv(DIR_GEN+r'/data/stock_index.csv',parse_dates = [1],index_col=[0,1])
    sr_rev_1 = pd.read_csv(DIR_GEN+r'/pnl/rev01.csv',parse_dates = [0],index_col=[0]).iloc[:,0]
    df_industry = pd.read_csv(DIR_GEN+'/data/stock_industry.csv',parse_dates = [1],index_col=1)
    df_industry = pd.pivot(df_industry.index,df_industry.SecuCode,df_industry.ZXF)
    df_industry = df_industry.fillna(np.nan)

    print( Name)

    df_Panel = getBaseData(df)

    df_is_comb1500 = df_Panel['value'].rank(axis=1,ascending=False) <= 1500
    df_is_comb1500 = df_is_comb1500.replace(False,np.nan)
    df_Panel = df_Panel.mul(df_is_comb1500,axis=0)

    df_close_pct = df_Panel['close'].pct_change()

    df_signal_rank_map = signal(df_Panel['close']
                                ,**signal_karg)

#    df_signal_rank_map = industry_renf(df_signal_rank_map
#                                       ,df_close_pct
#                                       ,df_industry
#                                       ,interval='Q'
#                                       )

    signal_sum_perday = abs(df_signal_rank_map).sum(axis=1)
    df_weight = df_signal_rank_map.div(signal_sum_perday.values,axis=0)

#    pdb.set_trace()

    df_close_pct_tomorrow = df_close_pct.shift(-1)

    df_return  = df_weight * df_close_pct_tomorrow


    print ('params:',params)
    params_str = json.dumps(params,indent=0)
    params_str = params_str[1:-1]

    msg_str = json.dumps(print_ret_msg(df_return,df_weight,df_close_pct_tomorrow,sr_rev_1).astype('str').str.slice(0,5).to_dict(),indent=0)[1:-1]

    sr_return_long = (df_close_pct_tomorrow*(df_weight[df_weight>0])).sum(1).cumsum()
    sr_return_short = (df_close_pct_tomorrow*(df_weight[df_weight<0])).sum(1).cumsum()

    fig = plot_top_return(df_return,df_Panel['close'],sr_return_long,sr_return_short)
#    fig.autofmt_xdate()

    fig.savefig(PathBackTestResult.format(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),Name), dpi=fig.dpi)






































































