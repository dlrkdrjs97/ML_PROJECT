import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
#from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
import gc
import os

merged_df = pd.read_csv('gvalue.csv')
len_train=307511
SEED = 7
# Re-separate into train and test
train_df = merged_df[:len_train]
test_df = merged_df[len_train:]
#del merged_df, app_test_df, app_train_df, bureau_df, bureau_balance_df, credit_card_df, pos_cash_df, prev_app_df
gc.collect()
""" Train the model """

target = train_df.pop('TARGET')

test_df.drop(columns='TARGET', inplace=True)

#---------------

print('data done')

data_train = train_df

data_test = test_df

 

data_train.fillna(-1, inplace=True)

data_test.fillna(-1, inplace=True)

#####################################################

data_train= data_train.replace(np.inf,0)
data_test= data_test.replace(np.inf,0)
############################################

cols = data_train.columns

 

ntrain = data_train.shape[0]

ntest = data_test.shape[0]

 

print(data_train.shape)

from sklearn.cross_validation import KFold

kf = KFold(data_train.shape[0], n_folds=5, shuffle=True, random_state=7)

NFOLDS = 5

x_train = np.array(data_train)

x_test = np.array(data_test)

y_train = np.array(target)

import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier,VotingClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import confusion_matrix
import xgboost as xgb
# This model uses XGB Classifier without any preprocessing process
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mplt
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib
from sklearn.preprocessing import normalize
import pickle

from sklearn.model_selection import train_test_split

import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
np.set_printoptions(threshold=np.nan)

for i, (train_index, test_index) in enumerate(kf):

    x_tr = x_train[train_index]

    y_tr = y_train[train_index]

    x_te = x_train[test_index]
    y_te = y_train[test_index]
# from https://www.kaggle.com/mmueller/stacking-starter?scriptVersionId=390867/code
class SklearnWrapper(object):
    def __init__(self, clf, seed=7, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        print("Training..")
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        print("Predicting..")
        return self.clf.predict_proba(x)


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        print("Training..")
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        print("Predicting..")
        return self.gbdt.predict(xgb.DMatrix(x))


def get_oof(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)[:,1]  # or [:,0]
        oof_test_skf[i, :] = clf.predict(x_test)[:,1]  # or [:,0]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
def get_oof_xgb(clf):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)  # or [:,0]
        oof_test_skf[i, :] = clf.predict(x_test)  # or [:,0]

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)
    
et_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.5,
    'max_depth': 10,
    'min_samples_leaf': 2,
}

rf_params = {
    'n_jobs': 16,
    'n_estimators': 100,
    'max_features': 0.2,
    'max_depth': 10,
    'min_samples_leaf': 2,
}

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.065,
    'objective': 'reg:linear',
    'max_depth': 9,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'auc',
    'nrounds': 350
}

cb_params = {
    'iterations':1000,
    'learning_rate':0.1,
    'depth':6,
    'l2_leaf_reg':40,
    'bootstrap_type':'Bernoulli',
    'subsample':0.7,
    'scale_pos_weight':5,
    'eval_metric':'AUC',
    'metric_period':50,
    'od_type':'Iter',
    'od_wait':45,
    'allow_writing_files':False    
}
xg = XgbWrapper(seed=SEED, params=xgb_params)
et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
cb = SklearnWrapper(clf=CatBoostClassifier, seed=SEED, params=cb_params)

print("xg..")
xg_oof_train, xg_oof_test = get_oof_xgb(xg)
print("et..")
et_oof_train, et_oof_test = get_oof(et)
print("rf..")
rf_oof_train, rf_oof_test = get_oof(rf)
print("cb..")
cb_oof_train, cb_oof_test = get_oof(cb)
x_train = np.concatenate((xg_oof_train, et_oof_train, rf_oof_train, cb_oof_train), axis=1)
x_test = np.concatenate((xg_oof_test, et_oof_test, rf_oof_test, cb_oof_test), axis=1)
np.save('x_train', x_train)
np.save('x_test', x_test)
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test)
subm = pd.read_csv('submission.csv')
xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'silent': 1,
    'subsample': 0.6,
    'learning_rate': 0.01,
    'objective': 'reg:linear',
    'max_depth': 5,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'auc',
}

print("xgb cv..")
res = xgb.cv(xgb_params, dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
             early_stopping_rounds=50, verbose_eval=10, show_stdv=True)
best_nrounds = res.shape[0] - 1

print("meta xgb train..")
gbdt = xgb.train(xgb_params, dtrain, best_nrounds)
fi = gbdt.predict(dtest)
fi = np.array(fi)
np.save('fi', fi)

subm['TARGET'] = fi
subm.to_csv('stack3_diff_data.csv', index=False)