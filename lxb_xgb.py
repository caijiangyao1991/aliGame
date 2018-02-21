<<<<<<< HEAD
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
pd.set_option('display.max_rows',None)


def getDesc4df(df):
    print ('the proceed  shape')
    print (df.shape)
    print ('the proceed  columns')
    print (df.columns)


def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print ( "confusion_matrix(left labels: y_true, up labels: y_pred):")
    print ("labels\t"),
    for i in range(len(labels)):
        print (labels[i], "\t"),
    print
    for i in range(len(conf_mat)):
        print (i, "\t"),
        for j in range(len(conf_mat[i])):
            print (conf_mat[i][j], '\t'),
        print
    print (float(conf_mat[1][1]) / (conf_mat[1][1] + conf_mat[1][0]))

def resultProb(dtest,prob):
    dtest['res'] =  dtest['predprob'].map(lambda x: 1 if x >= prob else 0)
    confusMatrix = metrics.classification_report(dtest[target].values, dtest['res'])
    print (confusMatrix)
    return confusMatrix


def modelfit_save(f, alg, dtrain, dtest, predictors, useTrainCV=False, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    trianMatrix = metrics.classification_report(dtrain[target].values, dtrain_predictions)
    print
    trianMatrix
    f.write("train _---------- 0.5\n")
    f.write(trianMatrix + '\n')

    print('ok')

    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    problist = [0.5, 0.4, 0.3, 0.25, 0.2]
    for eachProb in problist:
        f.write('test--------------' + str(eachProb) + ":\n")
        fuMatrix = resultProb(dtest, eachProb)
        f.write(fuMatrix + '\n')
    return dtest


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=False, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    trianMatrix = metrics.classification_report(dtrain[target].values, dtrain_predictions)
    print(trianMatrix)
    print('ok')
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    return dtest

def setXgbPara(n_estimators,max_depth,min_child_weight,gamma):
    #7 4 0.2
    xgb1 = XGBClassifier(
                learning_rate =0.1,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=84)
    return xgb1




def testResult(randomList, n_estimatorsList, max_depthList, min_child_weightList, gammaList):
    f = open('./outData/result_ddd.txt', 'w')

    for random in randomList:
        for n_estimators in n_estimatorsList:
            for depth in max_depthList:
                for child in min_child_weightList:
                    for gamma in gammaList:
                        writeStr = "\n" + str(random) + "," + str(n_estimators) + "," + str(depth) + "," + str(
                            child) + "," + str(gamma) + " :reslt\n"
                        f.write(writeStr)
                        predictors, split_train, split_test = genTestData(random)
                        xgb = setXgbPara(n_estimators, depth, child, gamma)
                        modelfit_save(f, xgb, split_train, split_test, predictors, useTrainCV=False)

    f.close()

if __name__ == '__main__':
    # TODO 训练集
    # 读取处理过的数据源
    train = pd.read_csv('./data/new_train_allfuture_reduce2.csv')
    getDesc4df(train)

    # 特征转换
    train['hos_fre'] = train['dummycount']
    train['pid'] = train['personId']
    # 删除无用的特征
    del train['dummycount']
    del train['personId']

    print('the proceed train shape')
    print(train.shape)
    print('the proceed train columns')
    print(train.columns)

    # 药品费用 添加后0.3达到5.1 estimetor=78 dpeth 6 4  gama0.0
    ypf = pd.read_csv('./data/ypf.csv')
    train = pd.merge(train, ypf, on='pid')
    # 加上药品报销比例
    train['ypclaimRatio'] = train['ypselfsum'] / train['ypsum']
    train['ypcavgRatio'] = train['ypselfavg'] / train['ypavg']
    # print ('添加药品与其报销费用')
    # print (train.shape)

    # 扩展总费用属性 自付情况
    feeSum = pd.read_csv('./data/allfee.csv')
    train = pd.merge(train, feeSum, on='pid')
    train['claimRatio'] = train['claimSum'] / train['sum']
    train['selfpy'] = train['sum'] - train['claimSum']
    # 所有报销费用平均值/所有费用平均值
    train['avgclaimRatio'] = train['claimSum.1'] / train['avg']

    # print ('添加总费用与报销费用')
    # print (train.shape)

    # 去掉不同的价格的计数会下降
    diffPrice = pd.read_csv('./data/diifpricecount.csv')
    train = pd.merge(train, diffPrice, on='pid', how='left')
    train['itemSum'] = train['itemSum'].map(lambda x: x if x > 0 else 0)
    # print ('添加不用价格药品数量')
    # print (train.shape)

    # add person level info 49
    plevel = pd.read_csv('./data/person_hos_fre_cnt.csv')
    plevel.columns = ['pid', 'a', 'b', 'c', 'd']
    train = pd.merge(train, plevel, on='pid', how='left')
    # print ('添加医院级别相关信息')
    # print (train.shape)

    # 添加重复住院天数提高0.3下的F1值到4.7   只添加重复住院天数F1A值最高提高到4.8
    duhos = pd.read_csv('./data/tmp_duphos.csv')
    train = pd.merge(train, duhos, on='pid', how='left')
    train['duHosNum'] = train['duHosNum'].map(lambda x: x if x > 0 else 1)
    # print ('添加一天内去过最多医院')
    # print (train.shape)

    maxcardpay = pd.read_csv('./data/maxcardpay.csv')
    train = pd.merge(train, maxcardpay, on='pid', how='left')

    # 测试疾病
    diseasdf = pd.read_csv('./data/disease.csv', encoding='gb18030')
    diseasdf = diseasdf[['pid', 'diseaseNum']]
    train = pd.merge(train, diseasdf, on='pid', how='left')
    # print ('添加几个人的疾病种类数')
    # print (train.shape)

    # 测试黑名单
    filterl = pd.read_csv('./data/filterlist.csv')
    # print (filterl.head())
    train = pd.merge(train, filterl, on='pid', how='left')
    train['isblack'] = train['isblack'].map(lambda x: x if x == 0 else 1)
    # print ('添加没有出现过任何欺诈的医院相关黑白名单')
    # print (train.shape)

    interDescDf = pd.read_csv('./data/dew_interval_desc.csv')
    train = pd.merge(train, interDescDf, on='pid', how='left')
    # print ('添加就诊间隔的描述信息 无平均值')
    # print (train.shape)

    detailtype = pd.read_csv('./data/person_hos_final.csv')
    detailtype.fillna(0, inplace=True)
    train = pd.merge(train, detailtype, on='pid', how='left')
    # print ('添加人去每家医院的次数信息')
    # print (train.shape)

    periodMax = pd.read_csv('./data/person_oneMonthMaxFreNew.csv')
    periodMax.fillna(0, inplace=True)
    train = pd.merge(train, periodMax, on='pid', how='left')
    # print ('31天去医院的最高频次数')
    # print (train.shape)

    periodMax = pd.read_csv('./data/person_intervelDaysCount.csv')
    train = pd.merge(train, periodMax, on='pid', how='left')
    # print ('间隔天数频次')
    # print (train.shape)

    # person_disease_disperse = pd.read_csv('./data/person_disease_disperse.csv', encoding='gb18030')
    # train = pd.merge(train, person_disease_disperse, on='pid', how='left')
    # print ('疾病类别')
    # print (train.shape)

    # person_holidays = pd.read_csv('./data/person_holidays.csv')
    # train = pd.merge(train, person_holidays, on='pid', how='left')
    # print ('节假日就医次数')
    # print (train.shape)
    #
    person_datediff = pd.read_csv('./data/person_datediff.csv')
    train = pd.merge(train, person_datediff, on='pid', how='left')
    # print ('住院间隔天数')


    # person_lastday = pd.read_csv('./data/person_lastday.csv')
    # train = pd.merge(train, person_lastday, on='pid', how='left')
    # print ('最后一次就诊在一年中的第几天')
    # print (train.shape)

    # person_firstday = pd.read_csv('./data/person_firstday.csv')
    # train = pd.merge(train, person_firstday, on='pid', how='left')
    # print ('第一次就诊在一年中的第几天')
    # print (train.shape)

    # person_week_fre = pd.read_csv('./data/person_week_fre.csv')
    # train = pd.merge(train, person_week_fre, on='pid', how='left')
    # print ('每周去医院次数')
    # print (train.shape)

    person_week_sumcost = pd.read_csv('./data/person_week_sumcost.csv')
    train = pd.merge(train, person_week_sumcost, on='pid', how='left')
    # print ('每周去医院花费')

    # person_week_hosfre = pd.read_csv('./data/person_week_hosfre.csv')
    # train = pd.merge(train, person_week_hosfre, on='pid', how='left')
    # print ('每周去医院个数')

    person_oneMonthSumCost = pd.read_csv('./data/person_oneMonthSumCost.csv')
    train = pd.merge(train, person_oneMonthSumCost, on='pid', how='left')
    # print ('每月最大花费')

    person_oneMontDrugCost = pd.read_csv('./data/person_oneMontDrugCost.csv')
    train = pd.merge(train, person_oneMontDrugCost, on='pid', how='left')
    # print ('每月最大药品花费')

    # person_oneMonthProgramCost = pd.read_csv('./data/person_oneMonthProgramCost.csv')
    # train = pd.merge(train, person_oneMonthProgramCost, on='pid', how='left')
    # print ('每月最大检查花费')

    # person_oneMonthSelfpayCost = pd.read_csv('./data/person_oneMonthSelfpayCost.csv')
    # train = pd.merge(train, person_oneMonthSelfpayCost, on='pid', how='left')
    # print ('每月最大自费花费')

    # person_oneMonthSumCost_min = pd.read_csv('./data/person_oneMonthSumCost_min.csv')
    # train = pd.merge(train, person_oneMonthSumCost_min, on='pid', how='left')
    # print('每月最小总花费')
    #
    # person_oneMonthDrugCost_min = pd.read_csv('./data/person_oneMonthDrugCost_min.csv')
    # train = pd.merge(train, person_oneMonthDrugCost_min, on='pid', how='left')

    print(train.head())

    print(train.shape)
    print(train.columns)

    # TODO 测试集
    test = pd.read_csv('./data/new_train_allfuture_reduce_t.csv')
    # print (test.shape)
    test['hos_fre'] = test['dummycount']
    del test['dummycount']
    del test['personId']

    # 药品费
    ypf_t = pd.read_csv('./data/ypf_t.csv')
    test = pd.merge(test, ypf_t, on='pid')
    test['ypclaimRatio'] = test['ypselfsum'] / test['ypsum']
    test['ypcavgRatio'] = test['ypselfavg'] / test['ypavg']

    # 扩展总费用属性"pid","bxsum","bxavg","sum","avg"
    feeSum_t = pd.read_csv('./data/allfee_t.csv')
    test = pd.merge(test, feeSum_t, on='pid')
    test['claimSum'] = test['bxsum']
    test['claimSum.1'] = test['bxavg']
    del test['bxsum']
    del test['bxavg']

    test['claimRatio'] = test['claimSum'] / test['sum']
    test['selfpy'] = test['sum'] - test['claimSum']
    # 去掉平均报销费用差0.1f1
    test['avgclaimRatio'] = test['claimSum.1'] / test['avg']

    plevel_t = pd.read_csv('./data/person_hos_fre_cnt_t.csv')
    plevel_t.columns = ['pid', 'a', 'b', 'c', 'd']
    test = pd.merge(test, plevel, on='pid', how='left')

    duhos_t = pd.read_csv('./data/tmp_duphos_t.csv')
    test = pd.merge(test, duhos_t, on='pid', how='left')
    test['duHosNum'] = test['duHosNum'].map(lambda x: x if x > 0 else 1)

    # diease
    diseasdf_t = pd.read_csv('./data/disease_t.csv', encoding='gb18030')
    diseasdf_t = diseasdf_t[['pid', 'diseaseNum']]
    test = pd.merge(test, diseasdf_t, on='pid', how='left')
    # test['diseaseNum']=test['diseaseNum'].map(lambda x: x if x > 0 else 0)

    # -----------------------添加的属性-----
    # 去掉不同的价格的计数会下降
    test = pd.merge(test, diffPrice, on='pid', how='left')
    test['itemSum'] = test['itemSum'].map(lambda x: x if x > 0 else 0)

    # 测试黑名单
    filterl_t = pd.read_csv('./data/filterlist_t.csv')
    # print (filterl_t.head())
    test = pd.merge(test, filterl_t, on='pid', how='left')
    test['isblack'] = test['isblack'].map(lambda x: x if x == 0 else 1)

    interDescDf = pd.read_csv('./data/dew_interval_desc_t.csv')
    test = pd.merge(test, interDescDf, on='pid', how='left')

    maxcardpay_t = pd.read_csv('./data/maxcardpay_t.csv')
    test = pd.merge(test, maxcardpay_t, on='pid', how='left')

    detailtype_t = pd.read_csv('./data/person_hos_t.csv')
    detailtype_t.fillna(0, inplace=True)
    test = pd.merge(test, detailtype_t, on='pid', how='left')

    periodMax_t = pd.read_csv('./data/person_oneMonthMaxFreNew_t.csv')
    periodMax_t.fillna(0, inplace=True)
    test = pd.merge(test, periodMax_t, on='pid', how='left')

    periodMax_t = pd.read_csv('./data/person_intervelDaysCount_t.csv')
    test = pd.merge(test, periodMax_t, on='pid', how='left')

    # person_disease_disperse_t = pd.read_csv('./data/person_disease_disperse_t.csv', encoding='gb18030')
    # test = pd.merge(test, person_disease_disperse_t, on='pid', how='left')

    # person_holidays_t = pd.read_csv('./data/person_holidays_t.csv')
    # test = pd.merge(test, person_holidays_t, on='pid', how='left')
    #
    person_datediff_t = pd.read_csv('./data/person_datediff_t.csv')
    test = pd.merge(test, person_datediff_t, on='pid', how='left')

    # person_lastday_t = pd.read_csv('./data/person_lastday_t.csv')
    # test = pd.merge(test, person_lastday_t, on='pid', how='left')

    # person_firstday_t = pd.read_csv('./data/person_firstday_t.csv')
    # test = pd.merge(test, person_firstday_t, on='pid', how='left')

    # person_week_fre_t = pd.read_csv('./data/person_week_fre_t.csv')
    # test = pd.merge(test, person_week_fre_t, on='pid', how='left')

    person_week_sumcost_t = pd.read_csv('./data/person_week_sumcost_t.csv')
    test = pd.merge(test, person_week_sumcost_t, on='pid', how='left')

    # person_week_hosfre_t = pd.read_csv('./data/person_week_hosfre_t.csv')
    # test = pd.merge(test, person_week_hosfre_t, on='pid', how='left')

    person_oneMonthSumCost_t = pd.read_csv('./data/person_oneMonthSumCost_t.csv')
    test = pd.merge(test, person_oneMonthSumCost_t, on='pid', how='left')

    person_oneMontDrugCost_t = pd.read_csv('./data/person_oneMontDrugCost_t.csv')
    test = pd.merge(test, person_oneMontDrugCost_t, on='pid', how='left')

    # person_oneMonthProgramCost_t = pd.read_csv('./data/person_oneMonthProgramCost_t.csv')
    # test = pd.merge(test, person_oneMonthProgramCost_t, on='pid', how='left')

    # person_oneMonthSelfpayCost_t = pd.read_csv('./data/person_oneMonthSelfpayCost_t.csv')
    # test = pd.merge(test, person_oneMonthSelfpayCost_t, on='pid', how='left')

    # person_oneMonthSumCost_min_t = pd.read_csv('./data/person_oneMonthSumCost_min_t.csv')
    # test = pd.merge(test, person_oneMonthSumCost_min_t, on='pid', how='left')
    #
    # person_oneMonthDrugCost_min_t = pd.read_csv('./data/person_oneMonthDrugCost_min_t.csv')
    # test = pd.merge(test, person_oneMonthDrugCost_min_t, on='pid', how='left')

    ID = 'pid'
    target = 'class'
    test.shape

    # TODO 变量处理
    def genTestData(random = 26):
        predictors = [x for x in train.columns if
                      x not in ['class', 'pid', 'personId'] and 'feature42' not in x and 'intervalDays_' not in x]
        print(len(predictors))
        X_train, X_test, y_train, y_test = train_test_split(train[predictors], train[target], test_size=0.2,
                                                            random_state=random)
        split_train = train.ix[X_train.index]
        split_test = train.ix[X_test.index]

        # 处理只保留讲个15天以内的频次
        addArr = []
        for each in train.columns:
            if 'intervalDays_' not in each:
                continue
            freq = each.split("_")[1]
            if int(freq) < 16:
                # print(each)
                addArr.append(each)
        # print (len(addArr))
        predictors.extend(addArr)
        print(len(predictors))
        return predictors, split_train, split_test


    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=126,
        max_depth=6,
        min_child_weight=4,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=84)

    predictors,split_train,split_test=genTestData()
    # testResult([26],[126,148], [5,6],[3,4],[0.1])

    rs=modelfit(xgb1, train, test, predictors,useTrainCV=False)

    # param_test1 = {
    #     # 'max_depth': range(4,7,1),
    #     # 'min_child_weight':[3,4,5]
    #     'gamma': [i / 10.0 for i in range(0, 5)]
    # }
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=148, max_depth=5,
    #                                             min_child_weight=3, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
    #                                             objective='binary:logistic',nthread=4,  scale_pos_weight=1,
    #                                             seed=85),
    #                     param_grid=param_test1, scoring='f1', n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # print(gsearch1.best_params_, gsearch1.best_score_)

    rs['class']=rs['predprob'].map(lambda x: 1 if x >= 0.2185 else 0)
    rs.head()
    finalrs=rs[['pid','class']]
    print(len(finalrs[finalrs['class']==1]))
    finalrs.head()
    test=pd.read_csv('./data/df_id_test.csv',header=None)
    test.shape
    test.columns=["pid"]
    test.head()
    dfr=pd.merge(test,finalrs,left_on='pid',right_on='pid',how='left')
    print(dfr.shape)
=======
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from matplotlib.pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import IsolationForest
pd.set_option('display.max_rows',None)


def getDesc4df(df):
    print ('the proceed  shape')
    print (df.shape)
    print ('the proceed  columns')
    print (df.columns)


def my_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print ( "confusion_matrix(left labels: y_true, up labels: y_pred):")
    print ("labels\t"),
    for i in range(len(labels)):
        print (labels[i], "\t"),
    print
    for i in range(len(conf_mat)):
        print (i, "\t"),
        for j in range(len(conf_mat[i])):
            print (conf_mat[i][j], '\t'),
        print
    print (float(conf_mat[1][1]) / (conf_mat[1][1] + conf_mat[1][0]))

def resultProb(dtest,prob):
    dtest['res'] =  dtest['predprob'].map(lambda x: 1 if x >= prob else 0)
    confusMatrix = metrics.classification_report(dtest[target].values, dtest['res'])
    print (confusMatrix)
    return confusMatrix


def modelfit_save(f, alg, dtrain, dtest, predictors, useTrainCV=False, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    print(feat_imp)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
    trianMatrix = metrics.classification_report(dtrain[target].values, dtrain_predictions)
    print
    trianMatrix
    f.write("train _---------- 0.5\n")
    f.write(trianMatrix + '\n')

    print('ok')

    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    problist = [0.5, 0.4, 0.3, 0.25, 0.2]
    for eachProb in problist:
        f.write('test--------------' + str(eachProb) + ":\n")
        fuMatrix = resultProb(dtest, eachProb)
        f.write(fuMatrix + '\n')
    return dtest


def modelfit(alg, dtrain, dtest, predictors, useTrainCV=False, cv_folds=5, early_stopping_rounds=100):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                          metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
        print(cvresult.shape[0])
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))

    trianMatrix = metrics.classification_report(dtrain[target].values, dtrain_predictions)
    print(trianMatrix)
    print('ok')
    dtest['predprob'] = alg.predict_proba(dtest[predictors])[:, 1]
    return dtest

def setXgbPara(n_estimators,max_depth,min_child_weight,gamma):
    #7 4 0.2
    xgb1 = XGBClassifier(
                learning_rate =0.1,
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                gamma=gamma,
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=84)
    return xgb1




def testResult(randomList, n_estimatorsList, max_depthList, min_child_weightList, gammaList):
    f = open('./outData/result_ddd.txt', 'w')

    for random in randomList:
        for n_estimators in n_estimatorsList:
            for depth in max_depthList:
                for child in min_child_weightList:
                    for gamma in gammaList:
                        writeStr = "\n" + str(random) + "," + str(n_estimators) + "," + str(depth) + "," + str(
                            child) + "," + str(gamma) + " :reslt\n"
                        f.write(writeStr)
                        predictors, split_train, split_test = genTestData(random)
                        xgb = setXgbPara(n_estimators, depth, child, gamma)
                        modelfit_save(f, xgb, split_train, split_test, predictors, useTrainCV=False)

    f.close()

if __name__ == '__main__':
    # TODO 训练集
    # 读取处理过的数据源
    train = pd.read_csv('./data/new_train_allfuture_reduce2.csv')
    getDesc4df(train)

    # 特征转换
    train['hos_fre'] = train['dummycount']
    train['pid'] = train['personId']
    # 删除无用的特征
    del train['dummycount']
    del train['personId']

    print('the proceed train shape')
    print(train.shape)
    print('the proceed train columns')
    print(train.columns)

    # 药品费用 添加后0.3达到5.1 estimetor=78 dpeth 6 4  gama0.0
    ypf = pd.read_csv('./data/ypf.csv')
    train = pd.merge(train, ypf, on='pid')
    # 加上药品报销比例
    train['ypclaimRatio'] = train['ypselfsum'] / train['ypsum']
    train['ypcavgRatio'] = train['ypselfavg'] / train['ypavg']
    # print ('添加药品与其报销费用')
    # print (train.shape)

    # 扩展总费用属性 自付情况
    feeSum = pd.read_csv('./data/allfee.csv')
    train = pd.merge(train, feeSum, on='pid')
    train['claimRatio'] = train['claimSum'] / train['sum']
    train['selfpy'] = train['sum'] - train['claimSum']
    # 所有报销费用平均值/所有费用平均值
    train['avgclaimRatio'] = train['claimSum.1'] / train['avg']

    # print ('添加总费用与报销费用')
    # print (train.shape)

    # 去掉不同的价格的计数会下降
    diffPrice = pd.read_csv('./data/diifpricecount.csv')
    train = pd.merge(train, diffPrice, on='pid', how='left')
    train['itemSum'] = train['itemSum'].map(lambda x: x if x > 0 else 0)
    # print ('添加不用价格药品数量')
    # print (train.shape)

    # add person level info 49
    plevel = pd.read_csv('./data/person_hos_fre_cnt.csv')
    plevel.columns = ['pid', 'a', 'b', 'c', 'd']
    train = pd.merge(train, plevel, on='pid', how='left')
    # print ('添加医院级别相关信息')
    # print (train.shape)

    # 添加重复住院天数提高0.3下的F1值到4.7   只添加重复住院天数F1A值最高提高到4.8
    duhos = pd.read_csv('./data/tmp_duphos.csv')
    train = pd.merge(train, duhos, on='pid', how='left')
    train['duHosNum'] = train['duHosNum'].map(lambda x: x if x > 0 else 1)
    # print ('添加一天内去过最多医院')
    # print (train.shape)

    maxcardpay = pd.read_csv('./data/maxcardpay.csv')
    train = pd.merge(train, maxcardpay, on='pid', how='left')

    # 测试疾病
    diseasdf = pd.read_csv('./data/disease.csv', encoding='gb18030')
    diseasdf = diseasdf[['pid', 'diseaseNum']]
    train = pd.merge(train, diseasdf, on='pid', how='left')
    # print ('添加几个人的疾病种类数')
    # print (train.shape)

    # 测试黑名单
    filterl = pd.read_csv('./data/filterlist.csv')
    # print (filterl.head())
    train = pd.merge(train, filterl, on='pid', how='left')
    train['isblack'] = train['isblack'].map(lambda x: x if x == 0 else 1)
    # print ('添加没有出现过任何欺诈的医院相关黑白名单')
    # print (train.shape)

    interDescDf = pd.read_csv('./data/dew_interval_desc.csv')
    train = pd.merge(train, interDescDf, on='pid', how='left')
    # print ('添加就诊间隔的描述信息 无平均值')
    # print (train.shape)

    detailtype = pd.read_csv('./data/person_hos_final.csv')
    detailtype.fillna(0, inplace=True)
    train = pd.merge(train, detailtype, on='pid', how='left')
    # print ('添加人去每家医院的次数信息')
    # print (train.shape)

    periodMax = pd.read_csv('./data/person_oneMonthMaxFreNew.csv')
    periodMax.fillna(0, inplace=True)
    train = pd.merge(train, periodMax, on='pid', how='left')
    # print ('31天去医院的最高频次数')
    # print (train.shape)

    periodMax = pd.read_csv('./data/person_intervelDaysCount.csv')
    train = pd.merge(train, periodMax, on='pid', how='left')
    # print ('间隔天数频次')
    # print (train.shape)

    # person_disease_disperse = pd.read_csv('./data/person_disease_disperse.csv', encoding='gb18030')
    # train = pd.merge(train, person_disease_disperse, on='pid', how='left')
    # print ('疾病类别')
    # print (train.shape)

    # person_holidays = pd.read_csv('./data/person_holidays.csv')
    # train = pd.merge(train, person_holidays, on='pid', how='left')
    # print ('节假日就医次数')
    # print (train.shape)
    #
    person_datediff = pd.read_csv('./data/person_datediff.csv')
    train = pd.merge(train, person_datediff, on='pid', how='left')
    # print ('住院间隔天数')


    # person_lastday = pd.read_csv('./data/person_lastday.csv')
    # train = pd.merge(train, person_lastday, on='pid', how='left')
    # print ('最后一次就诊在一年中的第几天')
    # print (train.shape)

    # person_firstday = pd.read_csv('./data/person_firstday.csv')
    # train = pd.merge(train, person_firstday, on='pid', how='left')
    # print ('第一次就诊在一年中的第几天')
    # print (train.shape)

    # person_week_fre = pd.read_csv('./data/person_week_fre.csv')
    # train = pd.merge(train, person_week_fre, on='pid', how='left')
    # print ('每周去医院次数')
    # print (train.shape)

    person_week_sumcost = pd.read_csv('./data/person_week_sumcost.csv')
    train = pd.merge(train, person_week_sumcost, on='pid', how='left')
    # print ('每周去医院花费')

    # person_week_hosfre = pd.read_csv('./data/person_week_hosfre.csv')
    # train = pd.merge(train, person_week_hosfre, on='pid', how='left')
    # print ('每周去医院个数')

    person_oneMonthSumCost = pd.read_csv('./data/person_oneMonthSumCost.csv')
    train = pd.merge(train, person_oneMonthSumCost, on='pid', how='left')
    # print ('每月最大花费')

    person_oneMontDrugCost = pd.read_csv('./data/person_oneMontDrugCost.csv')
    train = pd.merge(train, person_oneMontDrugCost, on='pid', how='left')
    # print ('每月最大药品花费')

    # person_oneMonthProgramCost = pd.read_csv('./data/person_oneMonthProgramCost.csv')
    # train = pd.merge(train, person_oneMonthProgramCost, on='pid', how='left')
    # print ('每月最大检查花费')

    # person_oneMonthSelfpayCost = pd.read_csv('./data/person_oneMonthSelfpayCost.csv')
    # train = pd.merge(train, person_oneMonthSelfpayCost, on='pid', how='left')
    # print ('每月最大自费花费')

    # person_oneMonthSumCost_min = pd.read_csv('./data/person_oneMonthSumCost_min.csv')
    # train = pd.merge(train, person_oneMonthSumCost_min, on='pid', how='left')
    # print('每月最小总花费')
    #
    # person_oneMonthDrugCost_min = pd.read_csv('./data/person_oneMonthDrugCost_min.csv')
    # train = pd.merge(train, person_oneMonthDrugCost_min, on='pid', how='left')

    print(train.head())

    print(train.shape)
    print(train.columns)

    # TODO 测试集
    test = pd.read_csv('./data/new_train_allfuture_reduce_t.csv')
    # print (test.shape)
    test['hos_fre'] = test['dummycount']
    del test['dummycount']
    del test['personId']

    # 药品费
    ypf_t = pd.read_csv('./data/ypf_t.csv')
    test = pd.merge(test, ypf_t, on='pid')
    test['ypclaimRatio'] = test['ypselfsum'] / test['ypsum']
    test['ypcavgRatio'] = test['ypselfavg'] / test['ypavg']

    # 扩展总费用属性"pid","bxsum","bxavg","sum","avg"
    feeSum_t = pd.read_csv('./data/allfee_t.csv')
    test = pd.merge(test, feeSum_t, on='pid')
    test['claimSum'] = test['bxsum']
    test['claimSum.1'] = test['bxavg']
    del test['bxsum']
    del test['bxavg']

    test['claimRatio'] = test['claimSum'] / test['sum']
    test['selfpy'] = test['sum'] - test['claimSum']
    # 去掉平均报销费用差0.1f1
    test['avgclaimRatio'] = test['claimSum.1'] / test['avg']

    plevel_t = pd.read_csv('./data/person_hos_fre_cnt_t.csv')
    plevel_t.columns = ['pid', 'a', 'b', 'c', 'd']
    test = pd.merge(test, plevel, on='pid', how='left')

    duhos_t = pd.read_csv('./data/tmp_duphos_t.csv')
    test = pd.merge(test, duhos_t, on='pid', how='left')
    test['duHosNum'] = test['duHosNum'].map(lambda x: x if x > 0 else 1)

    # diease
    diseasdf_t = pd.read_csv('./data/disease_t.csv', encoding='gb18030')
    diseasdf_t = diseasdf_t[['pid', 'diseaseNum']]
    test = pd.merge(test, diseasdf_t, on='pid', how='left')
    # test['diseaseNum']=test['diseaseNum'].map(lambda x: x if x > 0 else 0)

    # -----------------------添加的属性-----
    # 去掉不同的价格的计数会下降
    test = pd.merge(test, diffPrice, on='pid', how='left')
    test['itemSum'] = test['itemSum'].map(lambda x: x if x > 0 else 0)

    # 测试黑名单
    filterl_t = pd.read_csv('./data/filterlist_t.csv')
    # print (filterl_t.head())
    test = pd.merge(test, filterl_t, on='pid', how='left')
    test['isblack'] = test['isblack'].map(lambda x: x if x == 0 else 1)

    interDescDf = pd.read_csv('./data/dew_interval_desc_t.csv')
    test = pd.merge(test, interDescDf, on='pid', how='left')

    maxcardpay_t = pd.read_csv('./data/maxcardpay_t.csv')
    test = pd.merge(test, maxcardpay_t, on='pid', how='left')

    detailtype_t = pd.read_csv('./data/person_hos_t.csv')
    detailtype_t.fillna(0, inplace=True)
    test = pd.merge(test, detailtype_t, on='pid', how='left')

    periodMax_t = pd.read_csv('./data/person_oneMonthMaxFreNew_t.csv')
    periodMax_t.fillna(0, inplace=True)
    test = pd.merge(test, periodMax_t, on='pid', how='left')

    periodMax_t = pd.read_csv('./data/person_intervelDaysCount_t.csv')
    test = pd.merge(test, periodMax_t, on='pid', how='left')

    # person_disease_disperse_t = pd.read_csv('./data/person_disease_disperse_t.csv', encoding='gb18030')
    # test = pd.merge(test, person_disease_disperse_t, on='pid', how='left')

    # person_holidays_t = pd.read_csv('./data/person_holidays_t.csv')
    # test = pd.merge(test, person_holidays_t, on='pid', how='left')
    #
    person_datediff_t = pd.read_csv('./data/person_datediff_t.csv')
    test = pd.merge(test, person_datediff_t, on='pid', how='left')

    # person_lastday_t = pd.read_csv('./data/person_lastday_t.csv')
    # test = pd.merge(test, person_lastday_t, on='pid', how='left')

    # person_firstday_t = pd.read_csv('./data/person_firstday_t.csv')
    # test = pd.merge(test, person_firstday_t, on='pid', how='left')

    # person_week_fre_t = pd.read_csv('./data/person_week_fre_t.csv')
    # test = pd.merge(test, person_week_fre_t, on='pid', how='left')

    person_week_sumcost_t = pd.read_csv('./data/person_week_sumcost_t.csv')
    test = pd.merge(test, person_week_sumcost_t, on='pid', how='left')

    # person_week_hosfre_t = pd.read_csv('./data/person_week_hosfre_t.csv')
    # test = pd.merge(test, person_week_hosfre_t, on='pid', how='left')

    person_oneMonthSumCost_t = pd.read_csv('./data/person_oneMonthSumCost_t.csv')
    test = pd.merge(test, person_oneMonthSumCost_t, on='pid', how='left')

    person_oneMontDrugCost_t = pd.read_csv('./data/person_oneMontDrugCost_t.csv')
    test = pd.merge(test, person_oneMontDrugCost_t, on='pid', how='left')

    # person_oneMonthProgramCost_t = pd.read_csv('./data/person_oneMonthProgramCost_t.csv')
    # test = pd.merge(test, person_oneMonthProgramCost_t, on='pid', how='left')

    # person_oneMonthSelfpayCost_t = pd.read_csv('./data/person_oneMonthSelfpayCost_t.csv')
    # test = pd.merge(test, person_oneMonthSelfpayCost_t, on='pid', how='left')

    # person_oneMonthSumCost_min_t = pd.read_csv('./data/person_oneMonthSumCost_min_t.csv')
    # test = pd.merge(test, person_oneMonthSumCost_min_t, on='pid', how='left')
    #
    # person_oneMonthDrugCost_min_t = pd.read_csv('./data/person_oneMonthDrugCost_min_t.csv')
    # test = pd.merge(test, person_oneMonthDrugCost_min_t, on='pid', how='left')

    ID = 'pid'
    target = 'class'
    test.shape

    # TODO 变量处理
    def genTestData(random = 26):
        predictors = [x for x in train.columns if
                      x not in ['class', 'pid', 'personId'] and 'feature42' not in x and 'intervalDays_' not in x]
        print(len(predictors))
        X_train, X_test, y_train, y_test = train_test_split(train[predictors], train[target], test_size=0.2,
                                                            random_state=random)
        split_train = train.ix[X_train.index]
        split_test = train.ix[X_test.index]

        # 处理只保留讲个15天以内的频次
        addArr = []
        for each in train.columns:
            if 'intervalDays_' not in each:
                continue
            freq = each.split("_")[1]
            if int(freq) < 16:
                # print(each)
                addArr.append(each)
        # print (len(addArr))
        predictors.extend(addArr)
        print(len(predictors))
        return predictors, split_train, split_test


    xgb1 = XGBClassifier(
        learning_rate=0.1,
        n_estimators=126,
        max_depth=6,
        min_child_weight=4,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=84)

    predictors,split_train,split_test=genTestData()
    # testResult([26],[126,148], [5,6],[3,4],[0.1])

    rs=modelfit(xgb1, train, test, predictors,useTrainCV=False)

    # param_test1 = {
    #     # 'max_depth': range(4,7,1),
    #     # 'min_child_weight':[3,4,5]
    #     'gamma': [i / 10.0 for i in range(0, 5)]
    # }
    # gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=148, max_depth=5,
    #                                             min_child_weight=3, gamma=0.1, subsample=0.8, colsample_bytree=0.8,
    #                                             objective='binary:logistic',nthread=4,  scale_pos_weight=1,
    #                                             seed=85),
    #                     param_grid=param_test1, scoring='f1', n_jobs=4, iid=False, cv=5)
    # gsearch1.fit(train[predictors], train[target])
    # print(gsearch1.best_params_, gsearch1.best_score_)

    rs['class']=rs['predprob'].map(lambda x: 1 if x >= 0.2185 else 0)
    rs.head()
    finalrs=rs[['pid','class']]
    print(len(finalrs[finalrs['class']==1]))
    finalrs.head()
    test=pd.read_csv('./data/df_id_test.csv',header=None)
    test.shape
    test.columns=["pid"]
    test.head()
    dfr=pd.merge(test,finalrs,left_on='pid',right_on='pid',how='left')
    print(dfr.shape)
>>>>>>> a3e3b6a1e523fa340f88b8014234e2fd37d6ab2d
    dfr.to_csv('./outData/df_id_test_7.csv',index=False,header=False)