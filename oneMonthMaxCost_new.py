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

df_train = pd.read_csv('./rawData/tmp_train_new.csv',dtype={'pid':np.str})

print(df_train.head())
days = df_train[['pid','jtime','SUM(df_train.本次审批金额)','SUM(df_train.药品费发生金额)','SUM(df_train.检查费发生金额)','SUM(df_train.非账户支付金额)']]
days.columns = ['pid','jtime','sumCost','drugCost','programCost','selfpyCost']
days['jtime'] = pd.to_datetime(days['jtime'])
days.sort_values(by=['pid','jtime'], ascending=[False, True],inplace=True)
days['oneMonthAfer'] = pd.DataFrame(days['jtime']+pd.Timedelta('31 days'))
print(days.head())

pid = days['pid'].drop_duplicates()
print(len(pid))

# sumCost = []
# for id in pid:
#     person = days[days['pid']==id]
#     for i in range(len(person)):
#         atime = list(person['jtime'])[i]
#         btime = list(person['oneMonthAfer'])[i]
#         personSumCost = person[(person['jtime']>= atime) & (person['jtime']<= btime)]
#         personSumCost_1 = personSumCost['sumCost'].groupby(personSumCost['pid']).sum().reset_index()
#         sumCost.append(max(personSumCost_1['sumCost']))
#
# days['oneMonthSumCost'] = list(sumCost)
# days.sort_values(by=['pid','oneMonthSumCost'], ascending=[False, False],inplace=True)
# days_final = days.drop_duplicates(subset=['pid'], keep='first')
# days_final = days_final[['pid','oneMonthSumCost']]
# print(days_final.head())
# days_final.to_csv('./data/person_oneMonthSumCost.csv',index=False)

# drugCost = []
# for id in pid:
#     person = days[days['pid']==id]
#     for i in range(len(person)):
#         atime = list(person['jtime'])[i]
#         btime = list(person['oneMonthAfer'])[i]
#         personDrugCost = person[(person['jtime']>= atime) & (person['jtime']<= btime)]
#         personDrugCost_1 = personDrugCost['drugCost'].groupby(personDrugCost['pid']).sum().reset_index()
#         drugCost.append(max(personDrugCost_1['drugCost']))
#
# days['oneMonthDrugCost'] = list(drugCost)
# days.sort_values(by=['pid','oneMonthDrugCost'], ascending=[False, False],inplace=True)
# days_final = days.drop_duplicates(subset=['pid'], keep='first')
# days_final = days_final[['pid','oneMonthDrugCost']]
# print(days_final.head())
# days_final.to_csv('./data/person_oneMontDrugCost.csv',index=False)

#项目和非账户支付金额
programCost = []
selfpyCost = []
for id in pid:
    person = days[days['pid']==id]
    for i in range(len(person)):
        atime = list(person['jtime'])[i]
        btime = list(person['oneMonthAfer'])[i]
        personCost = person[(person['jtime']>= atime) & (person['jtime']<= btime)]

        personProgramCost = personCost['programCost'].groupby(personCost['pid']).sum().reset_index()
        personSelfpyCost = personCost['selfpyCost'].groupby(personCost['pid']).sum().reset_index()
        programCost.append(max(personProgramCost['programCost']))
        selfpyCost.append(max(personSelfpyCost['selfpyCost']))

days['oneMonthProgramCost'] = list(programCost)
days['oneMonthSelfpayCost'] = list(selfpyCost)
days_1 = days.sort_values(by=['pid','oneMonthProgramCost'], ascending=[False, False])
days_program = days_1.drop_duplicates(subset=['pid'], keep='first')
days_program = days_program[['pid','oneMonthProgramCost']]
print(days_program.head())
days_program.to_csv('./data/person_oneMonthProgramCost.csv',index=False)

days_2 = days.sort_values(by=['pid','oneMonthSelfpayCost'], ascending=[False, False])
days_selfpay = days_2.drop_duplicates(subset=['pid'], keep='first')
days_selfpay = days_selfpay[['pid','oneMonthSelfpayCost']]
print(days_selfpay.head())
days_selfpay.to_csv('./data/person_oneMonthSelfpayCost.csv',index=False)


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

df_train = pd.read_csv('./rawData/tmp_train_new.csv',dtype={'pid':np.str})

print(df_train.head())
days = df_train[['pid','jtime','SUM(df_train.本次审批金额)','SUM(df_train.药品费发生金额)','SUM(df_train.检查费发生金额)','SUM(df_train.非账户支付金额)']]
days.columns = ['pid','jtime','sumCost','drugCost','programCost','selfpyCost']
days['jtime'] = pd.to_datetime(days['jtime'])
days.sort_values(by=['pid','jtime'], ascending=[False, True],inplace=True)
days['oneMonthAfer'] = pd.DataFrame(days['jtime']+pd.Timedelta('31 days'))
print(days.head())

pid = days['pid'].drop_duplicates()
print(len(pid))

# sumCost = []
# for id in pid:
#     person = days[days['pid']==id]
#     for i in range(len(person)):
#         atime = list(person['jtime'])[i]
#         btime = list(person['oneMonthAfer'])[i]
#         personSumCost = person[(person['jtime']>= atime) & (person['jtime']<= btime)]
#         personSumCost_1 = personSumCost['sumCost'].groupby(personSumCost['pid']).sum().reset_index()
#         sumCost.append(max(personSumCost_1['sumCost']))
#
# days['oneMonthSumCost'] = list(sumCost)
# days.sort_values(by=['pid','oneMonthSumCost'], ascending=[False, False],inplace=True)
# days_final = days.drop_duplicates(subset=['pid'], keep='first')
# days_final = days_final[['pid','oneMonthSumCost']]
# print(days_final.head())
# days_final.to_csv('./data/person_oneMonthSumCost.csv',index=False)

# drugCost = []
# for id in pid:
#     person = days[days['pid']==id]
#     for i in range(len(person)):
#         atime = list(person['jtime'])[i]
#         btime = list(person['oneMonthAfer'])[i]
#         personDrugCost = person[(person['jtime']>= atime) & (person['jtime']<= btime)]
#         personDrugCost_1 = personDrugCost['drugCost'].groupby(personDrugCost['pid']).sum().reset_index()
#         drugCost.append(max(personDrugCost_1['drugCost']))
#
# days['oneMonthDrugCost'] = list(drugCost)
# days.sort_values(by=['pid','oneMonthDrugCost'], ascending=[False, False],inplace=True)
# days_final = days.drop_duplicates(subset=['pid'], keep='first')
# days_final = days_final[['pid','oneMonthDrugCost']]
# print(days_final.head())
# days_final.to_csv('./data/person_oneMontDrugCost.csv',index=False)

#项目和非账户支付金额
programCost = []
selfpyCost = []
for id in pid:
    person = days[days['pid']==id]
    for i in range(len(person)):
        atime = list(person['jtime'])[i]
        btime = list(person['oneMonthAfer'])[i]
        personCost = person[(person['jtime']>= atime) & (person['jtime']<= btime)]

        personProgramCost = personCost['programCost'].groupby(personCost['pid']).sum().reset_index()
        personSelfpyCost = personCost['selfpyCost'].groupby(personCost['pid']).sum().reset_index()
        programCost.append(max(personProgramCost['programCost']))
        selfpyCost.append(max(personSelfpyCost['selfpyCost']))

days['oneMonthProgramCost'] = list(programCost)
days['oneMonthSelfpayCost'] = list(selfpyCost)
days_1 = days.sort_values(by=['pid','oneMonthProgramCost'], ascending=[False, False])
days_program = days_1.drop_duplicates(subset=['pid'], keep='first')
days_program = days_program[['pid','oneMonthProgramCost']]
print(days_program.head())
days_program.to_csv('./data/person_oneMonthProgramCost.csv',index=False)

days_2 = days.sort_values(by=['pid','oneMonthSelfpayCost'], ascending=[False, False])
days_selfpay = days_2.drop_duplicates(subset=['pid'], keep='first')
days_selfpay = days_selfpay[['pid','oneMonthSelfpayCost']]
print(days_selfpay.head())
days_selfpay.to_csv('./data/person_oneMonthSelfpayCost.csv',index=False)


>>>>>>> a3e3b6a1e523fa340f88b8014234e2fd37d6ab2d
