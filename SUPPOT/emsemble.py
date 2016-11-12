# -*- coding: utf-8 -*-：
import pandas as pd
import time
import numpy as np
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.tree import export_graphviz
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

now = time.time()
IDset= pd.read_csv('F:/train/ID.csv') # 注意自己数据路径
ID = IDset.iloc[:,0].values
scaler = StandardScaler()
dataset = pd.read_csv('F:/train/TRAIN.csv') # 注意自己数据路径
dataset.loc[dataset['result'] == 1000, 'result'] = 1
dataset.loc[dataset['result'] ==1500, 'result'] =2
dataset.loc[dataset['result'] ==2000, 'result'] = 3
dataset['LIFTcost'] = dataset['LIFTcost'].fillna(dataset['LIFTcost'].median())
dataset['SCHOOL'] = dataset['SCHOOL'].fillna(dataset['SCHOOL'].median())
dataset['score'] = dataset['score'].fillna(dataset['score'].median())
dataset['WEEKEND_DAYS'] = dataset['WEEKEND_DAYS'].fillna(0)
dataset['WEEKENDCOST'] = dataset['WEEKENDCOST'].fillna(0)

xtrain = dataset.iloc[:,1:].values
labels = dataset.iloc[:,:1].values
scaler.fit(xtrain)
train = scaler.transform(xtrain)

tests = pd.read_csv('F:/train/TEST.csv') # 注意自己数据路径
tests['LIFTcost'] = tests['LIFTcost'].fillna(tests['LIFTcost'].median())
tests['score'] = tests['score'].fillna(1)
tests['SCHOOL'] = tests['SCHOOL'].fillna(tests['SCHOOL'].median())
tests['EAT_DAYS'] = tests['EAT_DAYS'].fillna(tests['EAT_DAYS'].median())
tests['TOTAL_EAT_COST'] = tests['TOTAL_EAT_COST'].fillna(tests['TOTAL_EAT_COST'].median())
tests['AVG_EATTING'] = tests['AVG_EATTING'].fillna(tests['AVG_EATTING'].median())
tests['WEEKEND_DAYS'] = tests['WEEKEND_DAYS'].fillna(0)
tests['WEEKENDCOST'] = tests['WEEKENDCOST'].fillna(0)
#test_id = range(len(tests))
xtest = tests.iloc[:,:].values
# print(dataset.describe())
# print(tests.describe())
# print(IDset.describe())
# print(labels.describe())
#
test = scaler.transform(xtest)

dataset.loc[dataset['result'] == 0, 'result'] = 1.305642633
dataset.loc[dataset['result'] == 1, 'result'] = 8.993252362
dataset.loc[dataset['result'] ==2, 'result'] =14.3311828
dataset.loc[dataset['result'] ==3, 'result'] = 318.82485876
sample_weight_last_ten = dataset.iloc[:,:1].values
sample_weight_last_ten = abs(np.random.randn(len(train)))
#拆分训练集和测试集
n_neighbors=5
algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [RandomForestClassifier(n_estimators = 30),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
            [neighbors.KNeighborsClassifier(n_neighbors),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]] ]
# # algorithms = [[GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]]]
# algorithms = [
#     [RandomForestClassifier(n_estimators = 30),["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]]]

# Initialize the cross validation folds
kf = KFold(len(train),n_folds=3, random_state=1)

predictions = []
for train_index, test_index in kf:
    train_target = labels
    full_test_predictions = []
    # for alg,pred in algorithms:
    #     # print algorithms[0][0]
    eclf1 = VotingClassifier(estimators=[('lr', algorithms[1][0]), ('rf', algorithms[0][0]),('KNN', algorithms[2][0])], voting='soft')
    feature_train,feature_test=train[train_index], train[test_index]
    target_train,target_test=labels[train_index], labels[test_index]
    sample_weight_T=sample_weight_last_ten[train_index]
    # Fit the algorithm on the training data.
    eclf1.fit(feature_train, target_train)
    train_predictions = eclf1.predict(feature_train)
    # Select and predict on the test fold.
    r = eclf1.score(feature_test, target_test)
    predictions.append(r)
    # test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
# print "训练"
# res = {}
# for i in labels:
#     res[i] = res.get(i, 0) + 1
# print([k for k in res.keys()])
# print([v for v in res.values()])


feature_train, feature_test, target_train, target_test,sample_weight_T,sample_weight_l = train_test_split(train, labels,sample_weight_last_ten, train_size=0.8, random_state=42)
r=f1_score(target_train, eclf1.predict(feature_train), average='macro')

print 'target_test %s' % r


print "预测结果"
res = {}
for i in train_predictions:
    res[i] = res.get(i, 0) + 1
print([k for k in res.keys()])
print([v for v in res.values()])


print predictions

ID_preds = eclf1.predict(test)

for  i in range(len(ID_preds)):
    if(ID_preds[i]==1):
        ID_preds[i]=1000
    else:
        if(ID_preds[i]==2):
            ID_preds[i] = 1500
        else:
            if (ID_preds[i] == 3):
                ID_preds[i] = 2000
print "the final result %s" % len(ID_preds)
res = {}
for i in ID_preds:
    res[i] = res.get(i, 0) + 1
print([k for k in res.keys()])
print([v for v in res.values()])
print "0 7571; 1000	740; 1500	465; 2000	354"
np.savetxt('F:/train/submission_xgb_MultiSoftmax.csv',np.c_[ID,ID_preds],
                delimiter=',',header='studentid,subsidy',comments='',fmt='%d')