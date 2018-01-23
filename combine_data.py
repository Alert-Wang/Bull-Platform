# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import sys,os
reload(sys)
sys.setdefaultencoding('utf-8')
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

from main import getDfFromDir,getTradDays

Data_Dir = '/dat/all/Equity/Wind/Daily/Stock/data'
Dir_Industry = '/dat/all/Equity/Wind/Daily/Industry/data'
Dir_index = '/dat/all/Equity/Wind/Daily/IndexEodPrices/data'

trade_days_path = '/dat5/src/lx/list/tradedays.csv'
start_day = '20030101'
end_day = '20171207'
output_path = '/home/wyy/Alpha/STOCK_DT/data/stock_history_daily_full.csv'
output_path_select = '/home/wyy/Alpha/STOCK_DT/data/stock_history_daily_select.csv'
output_path_industry = '/home/wyy/Alpha/STOCK_DT/data/stock_industry.csv'
output_path_index = '/home/wyy/Alpha/STOCK_DT/data/stock_index.csv'

DIR_GEN = '/home/wyy/Alpha/STOCK_DT/'


def combine_historyData(output_path,Data_Dir,colname=None):
    trade_days = getTradDays()
    trade_days = trade_days.loc[(trade_days >= start_day) & (trade_days <= end_day) ]
    df = pd.DataFrame()
    for datenum in trade_days:
        print datenum
        df = getDfFromDir(datenum,Data_Dir)
        if colname == None:
            pass
        else:
            df = df[colname]
        if os.path.isfile(output_path):
            with open(output_path,'a') as f:
                df.to_csv(f,header=False,index=False)
            f.close()
        else:
            df.to_csv(output_path,index=False,encoding='utf-8')


if __name__ == '__main__':

    colName = ['[1]SecuCode'
               ,'[2]TradingDay'
               ,'[4]OpenPrice'
               ,'[26]ClosePrice_BackFill'
               ,'[5]HighPrice'
               ,'[6]LowPrice'
               ,'[8]TurnoverVolume'
               ,'[7]ClosePrice'
               ,'[21]TotalShares']
    indexColName = ['[1]SecuCode'
               ,'[2]TradingDay'
               ,'[8]ClosePrice']

    combine_historyData(output_path_index,Dir_index)

































































