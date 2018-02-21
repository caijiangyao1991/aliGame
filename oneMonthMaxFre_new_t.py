<<<<<<< HEAD
# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from dataimputer import DataFrameImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
import os
pd.set_option('display.max_columns',None)

df_train = pd.read_csv('./rawData/tmp_train_new_t.csv',dtype={'pid':np.str})


days = df_train[['pid','jtime']]
days['jtime'] = pd.to_datetime(days['jtime'])
days.sort_values(by=['pid','jtime'], ascending=[False, True],inplace=True)

days['oneMonthAfer'] = pd.DataFrame(days['jtime']+pd.Timedelta('31 days'))

pid = days['pid'].drop_duplicates()
print(len(pid))
countfre = []

for id in pid:
    person = days[days['pid']==id]
    fre = pd.DataFrame()
    for i in range(len(person)):
        atime = list(person['jtime'])[i]
        btime = list(person['oneMonthAfer'])[i]
        personCount = person[(person['jtime']>= atime) & (person['jtime']<= btime)]
        personCount_1 = personCount.groupby(personCount['pid']).size().reset_index()
        personCount_1.columns = ['pid','oneMonthCount']
        countfre.append(max(personCount_1['oneMonthCount']))

days['oneMonthCount'] = list(countfre)
days.to_csv('./data/days_new_t.csv',index=False)

days.sort_values(by=['pid','oneMonthCount'], ascending=[False, False],inplace=True)
days_final = days.drop_duplicates(subset=['pid'], keep='first')
days_final = days_final[['pid','oneMonthCount']]
print(days_final.head())
days_final.to_csv('./data/person_oneMonthMaxFreNew_t.csv',index=False)
#
#

=======
# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from dataimputer import DataFrameImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.grid_search import GridSearchCV
import os
pd.set_option('display.max_columns',None)

df_train = pd.read_csv('./rawData/tmp_train_new_t.csv',dtype={'pid':np.str})


days = df_train[['pid','jtime']]
days['jtime'] = pd.to_datetime(days['jtime'])
days.sort_values(by=['pid','jtime'], ascending=[False, True],inplace=True)

days['oneMonthAfer'] = pd.DataFrame(days['jtime']+pd.Timedelta('31 days'))

pid = days['pid'].drop_duplicates()
print(len(pid))
countfre = []

for id in pid:
    person = days[days['pid']==id]
    fre = pd.DataFrame()
    for i in range(len(person)):
        atime = list(person['jtime'])[i]
        btime = list(person['oneMonthAfer'])[i]
        personCount = person[(person['jtime']>= atime) & (person['jtime']<= btime)]
        personCount_1 = personCount.groupby(personCount['pid']).size().reset_index()
        personCount_1.columns = ['pid','oneMonthCount']
        countfre.append(max(personCount_1['oneMonthCount']))

days['oneMonthCount'] = list(countfre)
days.to_csv('./data/days_new_t.csv',index=False)

days.sort_values(by=['pid','oneMonthCount'], ascending=[False, False],inplace=True)
days_final = days.drop_duplicates(subset=['pid'], keep='first')
days_final = days_final[['pid','oneMonthCount']]
print(days_final.head())
days_final.to_csv('./data/person_oneMonthMaxFreNew_t.csv',index=False)
#
#

>>>>>>> a3e3b6a1e523fa340f88b8014234e2fd37d6ab2d
