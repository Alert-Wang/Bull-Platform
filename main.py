# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import sys,os
#reload(sys)
#sys.setdefaultencoding('utf-8')
import pdb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import   RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
import sklearn
import collections
import graphviz
import pydot
import StringIO

Data_Dir = '/dat/all/Equity/Wind/Daily/Stock/data'
trade_days_path = '/dat5/src/lx/list/tradedays.csv'
start_day = '20040101'
end_day = '20161230'
output_path = '/home/wyy/Alpha/STOCK_DT/data/stock_history_daily.csv'


def getTradDays():
    sr = pd.read_csv(trade_days_path,header=None).astype('str')
    return pd.Series(sr[0])

'''
    功能函数，一般不会改动
'''
def getDfFromDir(date,_dir):
    datetime = time.strptime(date, '%Y%m%d')
    yearpath = os.path.join(_dir,str(datetime.tm_year))
    if os.path.isdir(yearpath):
        monthpath = os.path.join(yearpath,str(datetime.tm_mon).zfill(2))
        if os.path.isdir(monthpath):
            daypath = os.path.join(monthpath,str(datetime.tm_mday).zfill(2))
            if os.path.isdir(daypath):
                date_time = time.strftime('%Y-%m-%d',datetime)
                df = pd.read_csv(daypath+"/"+date_time+'.csv',
                                 sep = '|',
                                 encoding='utf-8'
                                 )
            else:
                raise AssertionError(daypath+'no exist!')
        else:
            raise AssertionError(monthpath+'no exist!')
    else:
        raise AssertionError(yearpath+'no exist!')
    return df

'''
    功能函数，将各个目录的数据合成一张表，只用一次，后续以后会用到
'''
def combine_historyData(output_path,Data_Dir):
    trade_days = getTradDays()
    trade_days = trade_days.loc[(trade_days >= start_day) & (trade_days <= end_day) ]
    df = pd.DataFrame()
    for datenum in trade_days:
        print datenum
        df = getDfFromDir(datenum,Data_Dir)
        #pdb.set_trace()
        if os.path.isfile(output_path):
            with open(output_path,'a') as f:
                df.to_csv(f,header=False,index=False)
            f.close()
        else:
            df.to_csv(output_path,index=False,encoding='utf-8')

    #df.to_csv(output_path,index=False)

'''
use to get feature, structure is to generate a dataframe column is 1,2,3,4,5 weekday
value is any indicator(price,volume)
'''
def getWeekDayDf(df_close,SCode):
#    SecuCode = '000001.SZ'
    sr_close = df_close[SCode]
    df_close = pd.DataFrame(sr_close.pct_change())

#    sr_close = sr_close.reset_index(level=1)
    df_close['week'] = sr_close.index.week
    df_close['weekday'] = sr_close.index.weekday + 1
    df_close['year'] = sr_close.index.year

    df_close['tag1'] = df_close['year'].astype('str') +'_' + df_close['week'].astype('str').str.zfill(2)

    df_pivot = pd.pivot_table(df_close, values=SCode, index=['tag1'],columns=['weekday'], aggfunc='mean')

    df_pivot = df_pivot.sort_index()
    df_pivot['EOD_sum'] = df_pivot.sum(axis=1)
    df_pivot['EOD_std'] = df_pivot.std(axis=1)

    df_pivot['label'] = df_pivot[2].shift(-1)
#    df_pivot['label'] = df_pivot.sum(axis=1).shift(-1)
    return df_pivot

def ger_rolling(sr,window=5):

    l_ = range(1,window+1)
    l_.reverse()
    df = pd.DataFrame()
    for i in xrange(len(sr)):
        i = i
#        if i%1000 ==0:
#            print i
        if i <= window:
            if i == 0:
                s_ = pd.Series(window*[np.nan])
                s_.index = list(range(1,window+1))

            else:
                s_ = pd.Series(sr.iloc[0:i])
                s_.index = l_[-i:]

                for j in l_[:-i]:
                    s_[j] = np.nan

            s_.name = sr.iloc[0:i+1].index[-1]
            df = df.append(s_)

        else:
            s_ = pd.Series(sr.iloc[i-window:i])
            s_.index = l_

            s_.name = sr.iloc[i-window:i+1].index[-1]
            df = df.append(s_)

    return df

def add_feature(df):
    df_add = pd.DataFrame()
    df_add['sum'] = df.sum(axis=1)

#    df_add['diff_1'] = df.diff(1).sum(axis=1)
#    df_add['diff_2'] = df.diff(2).sum(axis=1)

#    df_add['5dr'] = - df_add['sum']
#    df_add['sum_bool'] = df_add['sum'] > 0
#    df_add['std'] = df.std(axis=1)
#    df_add['skew'] = df.skew(axis=1)
#    df_add['kurt'] = df.kurt(axis=1)
    df_add['max'] = df.max(axis=1)
    df_add['min'] = df.min(axis=1)

    df_add['range_0'] = abs(df_add['max']) + abs(df_add['min'])
    df_add['range_1'] = abs(df_add['max'] - df_add['min'])


    df_add['weekday'] = df.index.weekday + 1
    df_add['monday'] = df_add['weekday'] == 1
#    df_add['tuseday'] = df_add['weekday'] == 2
#    df_add['thirsday'] = df_add['weekday'] == 3
#    df_add['wendsday'] = df_add['weekday'] == 4
#    df_add['friday'] = df_add['weekday'] == 5
    return df_add


def getNearDayDf(df_close,SCode):
#    SecuCode = '000001.SZ'
    sr_close = df_close[SCode]
    df_close = pd.DataFrame(sr_close.pct_change())

    df_pivot = ger_rolling(df_close[SCode])

    df_pivot = df_pivot.sort_index(axis=1,ascending=False)
    df_feature_add = add_feature(df_pivot)

    df_feature = pd.concat([df_pivot,df_feature_add],axis=1)

    df_feature['label'] = df_feature[1].shift(-1)#
#    df_pivot['label'] = df_pivot.sum(axis=1).shif-t(-1)
    return df_feature



def get_feat_importance(X_train,X_test,columns):
    train_data

    rdf = RandomForestClassifier(random_state=3
#                                 ,max_depth=6
                                 , n_estimators=100
#                                 , max_features=4
                                 ,n_jobs=8
#                                 ,class_weight="balanced"

                                 )
    rdf.fit(X_train,y_train)
    p = rdf.feature_importances_
    df_impt = pd.DataFrame(zip(columns,p),columns=['featrue','importance'])
    df_impt = df_impt.sort_values(by=['importance'],ascending=[0])


    return df_impt

def ger_gridsearch_df(model,X_train,y_train):
    parameters = {'max_depth':range(5,8,12)
                  ,'max_features':range(4,6)
                  ,'min_samples_split':[0.002,0.01,0.005]
                  ,'min_samples_leaf':[0.0002,0.0005,0.001]
                  }
    clf = GridSearchCV(model, parameters,verbose=5)
    clf.fit(X_train,y_train)
#    pdb.set_trace()
    df_gs = pd.DataFrame(clf.grid_scores_  )
    df_gs = df_gs.sort_values('mean_validation_score',ascending=False)
    return df_gs



if __name__ == '__main__':
    DIR_GEN = '/home/wyy/Alpha/STOCK_DT/'

    df = pd.read_csv(r'/home/wyy/Alpha/STOCK_DT/data/stock_history_daily.csv')
    #pdb.set_trace()


    df = df.drop_duplicates(subset=['[2]TradingDay','[1]SecuCode'])
    df['[2]TradingDay'] = pd.to_datetime(df['[2]TradingDay'].astype('str').str.slice(0,8))
    #df.sort_index(inplace=True)
#    pdb.set_trace()
    df = df.set_index(['[2]TradingDay','[1]SecuCode'],verify_integrity=False)

#    df_open = df['[4]OpenPrice']
#    df_open = df_open.unstack(level=-1)
#    df_open = df_open.dropna(axis=0,how='all')
#
#
#    df_close_bf = df['[26]ClosePrice_BackFill']
#    df_close_bf = df_close_bf.unstack(level=-1)
#    df_close_bf = df_close_bf.dropna(axis=0,how='all')
#
#    df_high = df['[5]HighPrice']
#    df_high = df_high.unstack(level=-1)
#    df_high = df_high.dropna(axis=0,how='all')
#
#    df_low = df['[6]LowPrice']
#    df_low = df_low.unstack(level=-1)
#    df_low = df_low.dropna(axis=0,how='all')
#
#    df_volume = df['[8]TurnoverVolume']
#    df_volume = df_volume.unstack(level=-1)
#    df_volume = df_volume.dropna(axis=0,how='all')

    df_close = df['[26]ClosePrice_BackFill']
    df_close = df_close.unstack(level=-1)
    df_close = df_close.dropna(axis=0,how='all')

    df_TotalShares = df['[21]TotalShares']
    df_TotalShares = df_TotalShares.unstack(level=-1)
    df_TotalShares = df_TotalShares.dropna(axis=0,how='all')

#----------------------------取comb1500,粗略取代码，顾及时间完整性-----------------------
    df_TotalValue = df_close * df_TotalShares
    df_is_comb1500 = df_TotalValue.rank(axis=1)
    df_is_comb1500 = df_is_comb1500 >=1500

#    df_close_decomb = df_is_comb1500*df_close
#    df_close_decomb = df_close_decomb.dropna(axis=1,how='all')
    secode_comb1500 = df_is_comb1500.dropna(axis=1,how='all').columns
    df_close = df_close[secode_comb1500]

#----------------------------取comb1500,粗略取代码，顾及时间完整性-----------------------


# =============================================================================
#     SecuCode = '000001.SZ'
#     sr_close = df_close[SecuCode]
#     df_close = pd.DataFrame(sr_close.pct_change())
#
# #    sr_close = sr_close.reset_index(level=1)
#     df_close['week'] = sr_close.index.week
#     df_close['weekday'] = sr_close.index.weekday + 1
#     df_close['year'] = sr_close.index.year
#
#     df_close['tag1'] = df_close['year'].astype('str') +'_' + df_close['week'].astype('str').str.zfill(2)
#
#     df_pivot = pd.pivot_table(df_close, values=SecuCode, index=['tag1'],columns=['weekday'], aggfunc='mean')
#
#     df_pivot = df_pivot.sort_index()
#
#     df_pivot['label'] = df_pivot[2]
# =============================================================================

#    SecuCode = '600613.SH'
    df_feature = pd.DataFrame()
    df_result = pd.DataFrame()

    SCodes = list(df.index.levels[1].unique())
    for i,code in enumerate(SCodes[:]):

#        SecuCode = '000001.SZ'
        print 15*'-',i,'-',code,15*'-'
        df_pivot = getNearDayDf(df_close,code)
#        df_feature = df_feature.append(df_pivot)
        df_feature = df_pivot.copy()

        df_feature = df_feature.dropna(how='all',subset=[1,2,3,4,5])
        df_feature = df_feature.drop([2,3,5],axis=1)
        df_feature = df_feature.fillna(0)

        sr_label_pct = df_feature['label'].copy()
        sr_label = df_feature.pop('label') >= 0

        if len(df_feature.index) <=200:
            continue

        train_data = df_feature.values
        train_label = np.array(sr_label)

        X_train,X_test, y_train, y_test = \
        train_test_split(train_data,train_label,test_size=0.25, random_state=5)

        dt_stump = DecisionTreeClassifier(
                                           splitter = 'best'
#                                           ,max_depth= X_train.shape[1]*1.5
                                           ,max_depth= X_train.shape[1]
                                           ,max_features = X_train.shape[1]
                                           ,min_samples_split = int(X_train.shape[0] * 0.03) + 1
                                           ,min_samples_leaf = int(X_train.shape[0] * 0.015) + 1
                                           ,max_leaf_nodes  = 15
#                                           ,max_leaf_nodes  =0
                                           ,criterion = 'entropy'
#                                           ,min_impurity_decrease = 0.075
#                                           ,min_impurity_split  = 0.075
                                           ,random_state = 1
#                                           ,class_weight='balanced'
                                           ,class_weight={0: 1, 1: 1.2}
                                           )
        dt_stump.fit(X_train,y_train)

        score_test = dt_stump.score(X_test,y_test)
        score_train = dt_stump.score(X_train,y_train)
        dt_stump_importance = dt_stump.feature_importances_

        y_pred = dt_stump.predict(X_test)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        print 'score_train:',score_train,'score_test:',score_test,'TN-ratio:',y_test.mean(),'Gain:',score_test - y_test.mean()
        print 'cnf_matrix:'
        print cnf_matrix

#----------------------------输出特征重要性-----------------------
        print 'dt importance:'
        df_impt_dt = pd.DataFrame(zip(df_feature.columns,dt_stump_importance),columns=['featrue','importance'])
        df_impt_dt = df_impt_dt.sort_values(by=['importance'],ascending=[0])
        print df_impt_dt

#        y_prob = dt_stump.predict_proba(X_test)

        print 'rdf importance martix:'
        df_impt_rdf = get_feat_importance(X_train,X_test,df_feature.columns)
        print df_impt_rdf
#----------------------------输出特征重要性-----------------------

        sr =pd.Series()
        sr.name = code
        sr['score_train'] = score_train
        sr['score_test'] = score_test
        sr['erro_rate'] = cnf_matrix[1,1]/float(cnf_matrix[0,1] + cnf_matrix[1,1])
        sr['rdf_featture_importance'] = collections.OrderedDict(zip(df_impt_rdf['featrue'],df_impt_rdf['importance']))
        sr['TP-ratio'] = y_test.mean()
        sr['Gain'] = score_test - y_test.mean()

        df_result = df_result.append(sr)

#----------------------------DT作图-----------------------
        dotfile = StringIO.StringIO()
        dot_data = sklearn.tree.export_graphviz(dt_stump
                                             ,out_file = dotfile
                                             ,feature_names=df_feature.columns
                                             ,class_names=['0','1']
                                             ,filled=True
                                             ,rounded=True
                                             ,leaves_parallel =True
                                             ,impurity=True
                                             ,special_characters=True
                                             )
#        sklearn.tree.export_graphviz(dt_stump, out_file=dotfile)
        pydot_obj = pydot.graph_from_dot_data(dotfile.getvalue())
        pydot_obj.write_png(DIR_GEN+'figure/dt_png/'+code+'.png')


#        dot_file = open(DIR_GEN+'figure/dt_dot/'+code+'.dot')
#
#        graph = graphviz.Source(dot_file)
#
#        os.system("dot -Tpng "+ DIR_GEN+'figure/dt_dot/'+code+'.dot'+ ' -o ' + DIR_GEN+'figure/dt_pdf/'+code+'.png')
#
#        dot_file.close()
##        pdb.set_trace()
#        graph.render(DIR_GEN+'figure/dt_pdf/'+code+'.pdf')
#----------------------------DT作图-----------------------

#    df_result = df_result.sort_values(by=['erro_rate','score_test'],ascending=[0,0])
    df_result = df_result.sort_values(by=['erro_rate'],ascending=[0])
#    df_result = df_result.sort_values(by=['score_test'],ascending=[0])
    print df_result
    pdb.set_trace()


#        df_gs = ger_gridsearch_df(dt_stump,X_train,y_train)





#    df_close = df_close.set_index(['tag1','weekday'])
#    df_close = df_close[SecuCode]
#
#
#    sr_close.index = (sr_close.index.weekday) + 1


    #combine_historyData(output_path,Data_Dir)

































































