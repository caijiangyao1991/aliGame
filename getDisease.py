<<<<<<< HEAD
# coding=utf-8
import pandas as pd
import numpy as np


df_train = pd.read_csv('./rawData/df_train.csv')
#去除挂号费用
df_train = df_train[(df_train['出院诊断病种名称'] !='挂号') & (df_train['出院诊断病种名称'] !='门特挂号')]
def get_splite_char(x):
    res = set()
    for char in x:
        if ('\u4e00' <= char <= '\u9fff') or ('\u0030' <= char<='\u0039') or ('\u0041' <= char<='\u005a') or ('\u0061' <= char<='\u007a'): #判别中文，字母，或数字
            continue
        else:
            res.add(char)
    res = "||".join(res)
    return res
df_train['出院诊断病种名称'] = df_train['出院诊断病种名称'].astype(str)
# df_train["split_char"] = df_train.出院诊断病种名称.apply(get_splite_char)
# split_char = set("||".join(df_train.split_char.values).split("||"))

# split_by_hand = set([' ',';','、','，','。','；',':','*'])
import re
# comp = re.compile(r"[,*、.-。，:;\[/\] ? —；）（\(@+ ]+")

comp = re.compile(r"[ \[/\] ; 、。；? : ／ ? - = ` / ,  ·  ：＼ ，’.  @ ；— * <  。^ 、…  ; ‘ + \( \) （）【】0 00 000 000? 2 ]+")
df_train["new"] = df_train.出院诊断病种名称.apply(lambda x: re.split(comp,x))
# print(df_train["new"])
df_train['mainDisease'] = df_train["new"].map(lambda x: ','.join(x).split(',')[0])
df_train['mainDisease'].replace('nan', np.nan,inplace=True)

df_train_disease = df_train[['个人编码','mainDisease','顺序号']]
df_train_disease.to_csv('df_train_disease.csv')


disease = pd.read_csv('df_train_disease.csv',encoding='gb18030')
#TODO 计算某个人患病最多的疾病
diseaseNum=disease[['个人编码', 'mainDisease','顺序号']].groupby(by=['个人编码','mainDisease']).count().reset_index() #均值标准差最大最小值都求出来了
# print(diseaseNum.head())

diseaseNum.sort_values(by=['个人编码','顺序号'],ascending=[False,False],inplace=True)
# print(diseaseNum.head(10))
diseaseNum_1 = diseaseNum.drop_duplicates(subset=['个人编码'],keep='first')
diseaseNum_1.rename(columns={'mainDisease':'mainDisease1','顺序号':'fre1'},inplace=True)
print(diseaseNum_1.head())

#TODO 计算某个人患病第二多的疾病
flag = diseaseNum_1.copy()
flag['flag']=1
diseaseNum_2 = pd.merge(diseaseNum,flag, left_on=['个人编码','mainDisease'],right_on=['个人编码','mainDisease1'],how='left' )
diseaseNum_2.drop(['mainDisease1','fre1'],axis='columns',inplace=True)
diseaseNum_2=diseaseNum_2[diseaseNum_2['flag'] != 1]
diseaseNum_2 = diseaseNum_2.drop_duplicates(subset=['个人编码'],keep='first')
diseaseNum_2.drop(['flag'],axis='columns',inplace=True)
diseaseNum_2.rename(columns={'mainDisease':'mainDisease2','顺序号':'fre2'},inplace=True)
print(diseaseNum_2.head())

#TODO 将同一个人患的所有主诊断拼接
from functools import reduce
disease['mainDisease'] = disease['mainDisease'].astype(str)
def flat(df):
    return reduce(lambda x,y: x + ',' + y , df)
items = disease['mainDisease'].groupby(disease['个人编码']).apply(flat).reset_index()
items['mainDisease'] = items['mainDisease'].map(lambda x : x.replace('nan',''))
items['newMainDisease'] = items['mainDisease'].map(lambda x:  ','.join(filter(None,set(x.split(",")))))
items['diseaseNum'] = items['newMainDisease'].map(lambda x : len(x.split(',')))
items = items[['个人编码','diseaseNum']]

#连接
diseaseFinal = pd.merge(diseaseNum_1,diseaseNum_2, left_on='个人编码',right_on='个人编码')
diseaseFinal_1 = pd.merge(diseaseFinal,items, left_on='个人编码',right_on='个人编码')

print(diseaseFinal_1.count().apply(lambda x: float(diseaseFinal_1.shape[0]-x)/diseaseFinal_1.shape[0]))
print(diseaseFinal_1.sort_values(by=['diseaseNum']))
diseaseFinal_1.to_csv('./data/disease.csv',index=False)
=======
# coding=utf-8
import pandas as pd
import numpy as np


df_train = pd.read_csv('./rawData/df_train.csv')
#去除挂号费用
df_train = df_train[(df_train['出院诊断病种名称'] !='挂号') & (df_train['出院诊断病种名称'] !='门特挂号')]
def get_splite_char(x):
    res = set()
    for char in x:
        if ('\u4e00' <= char <= '\u9fff') or ('\u0030' <= char<='\u0039') or ('\u0041' <= char<='\u005a') or ('\u0061' <= char<='\u007a'): #判别中文，字母，或数字
            continue
        else:
            res.add(char)
    res = "||".join(res)
    return res
df_train['出院诊断病种名称'] = df_train['出院诊断病种名称'].astype(str)
# df_train["split_char"] = df_train.出院诊断病种名称.apply(get_splite_char)
# split_char = set("||".join(df_train.split_char.values).split("||"))

# split_by_hand = set([' ',';','、','，','。','；',':','*'])
import re
# comp = re.compile(r"[,*、.-。，:;\[/\] ? —；）（\(@+ ]+")

comp = re.compile(r"[ \[/\] ; 、。；? : ／ ? - = ` / ,  ·  ：＼ ，’.  @ ；— * <  。^ 、…  ; ‘ + \( \) （）【】0 00 000 000? 2 ]+")
df_train["new"] = df_train.出院诊断病种名称.apply(lambda x: re.split(comp,x))
# print(df_train["new"])
df_train['mainDisease'] = df_train["new"].map(lambda x: ','.join(x).split(',')[0])
df_train['mainDisease'].replace('nan', np.nan,inplace=True)

df_train_disease = df_train[['个人编码','mainDisease','顺序号']]
df_train_disease.to_csv('df_train_disease.csv')


disease = pd.read_csv('df_train_disease.csv',encoding='gb18030')
#TODO 计算某个人患病最多的疾病
diseaseNum=disease[['个人编码', 'mainDisease','顺序号']].groupby(by=['个人编码','mainDisease']).count().reset_index() #均值标准差最大最小值都求出来了
# print(diseaseNum.head())

diseaseNum.sort_values(by=['个人编码','顺序号'],ascending=[False,False],inplace=True)
# print(diseaseNum.head(10))
diseaseNum_1 = diseaseNum.drop_duplicates(subset=['个人编码'],keep='first')
diseaseNum_1.rename(columns={'mainDisease':'mainDisease1','顺序号':'fre1'},inplace=True)
print(diseaseNum_1.head())

#TODO 计算某个人患病第二多的疾病
flag = diseaseNum_1.copy()
flag['flag']=1
diseaseNum_2 = pd.merge(diseaseNum,flag, left_on=['个人编码','mainDisease'],right_on=['个人编码','mainDisease1'],how='left' )
diseaseNum_2.drop(['mainDisease1','fre1'],axis='columns',inplace=True)
diseaseNum_2=diseaseNum_2[diseaseNum_2['flag'] != 1]
diseaseNum_2 = diseaseNum_2.drop_duplicates(subset=['个人编码'],keep='first')
diseaseNum_2.drop(['flag'],axis='columns',inplace=True)
diseaseNum_2.rename(columns={'mainDisease':'mainDisease2','顺序号':'fre2'},inplace=True)
print(diseaseNum_2.head())

#TODO 将同一个人患的所有主诊断拼接
from functools import reduce
disease['mainDisease'] = disease['mainDisease'].astype(str)
def flat(df):
    return reduce(lambda x,y: x + ',' + y , df)
items = disease['mainDisease'].groupby(disease['个人编码']).apply(flat).reset_index()
items['mainDisease'] = items['mainDisease'].map(lambda x : x.replace('nan',''))
items['newMainDisease'] = items['mainDisease'].map(lambda x:  ','.join(filter(None,set(x.split(",")))))
items['diseaseNum'] = items['newMainDisease'].map(lambda x : len(x.split(',')))
items = items[['个人编码','diseaseNum']]

#连接
diseaseFinal = pd.merge(diseaseNum_1,diseaseNum_2, left_on='个人编码',right_on='个人编码')
diseaseFinal_1 = pd.merge(diseaseFinal,items, left_on='个人编码',right_on='个人编码')

print(diseaseFinal_1.count().apply(lambda x: float(diseaseFinal_1.shape[0]-x)/diseaseFinal_1.shape[0]))
print(diseaseFinal_1.sort_values(by=['diseaseNum']))
diseaseFinal_1.to_csv('./data/disease.csv',index=False)
>>>>>>> a3e3b6a1e523fa340f88b8014234e2fd37d6ab2d
