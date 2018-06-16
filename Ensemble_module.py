

## ENSEMBLE MODULE



## IMPORT EXTERN MODULE
from Wrapper import SklearnWrapper, XgbWrapper


class ensembletrain:
    def setdata(self, merged_df, len_train):
        self.merged_df = merged_df
        self.len_train = len_train
        self.SEED = 7
        self.train_df = self.merged_df[:self.len_train]
        self.test_df = self.merged_df[self.len_train:]
        
    def set_train(self):
        self.et_params = {
            'n_jobs': 16,
            'n_estimators': 100,
            'max_features': 0.5,
            'max_depth': 10,
            'min_samples_leaf': 2,
        }

        self.rf_params = {
            'n_jobs': 16,
            'n_estimators': 100,
            'max_features': 0.2,
            'max_depth': 10,
            'min_samples_leaf': 2,
        }

        self.xgb_params = {
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

        self.cb_params = {
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
        self.target = self.train_df.pop('TARGET')
        self.test_df.drop(columns='TARGET', inplace=True)
        print('data done')
        self.data_train = self.train_df
        self.data_test = self.test_df
        self.data_train.fillna(-1, inplace=True)
        self.data_test.fillna(-1, inplace=True)
        self.data_train= self.data_train.replace(np.inf,0)
        self.data_test= self.data_test.replace(np.inf,0)
        self.cols = self.data_train.columns
        self.ntrain = self.data_train.shape[0]
        self.ntest = self.data_test.shape[0]
        print(self.data_train.shape)
        from sklearn.cross_validation import KFold
        self.kf = KFold(self.data_train.shape[0], n_folds=5, shuffle=True, random_state=7)
        NFOLDS = 5
        self.x_train = np.array(self.data_train)
        self.x_test = np.array(self.data_test)
        self.y_train = np.array(self.target)
        
        for i, (train_index, test_index) in enumerate(self.kf):

            self.x_tr = self.x_train[train_index]
            self.y_tr = self.y_train[train_index]

            self.x_te = self.x_train[test_index]
            self.y_te = self.y_train[test_index]
            
        self.xg = XgbWrapper(seed=SEED, params=self.xgb_params)
        self.et = SklearnWrapper(clf=ExtraTreesClassifier, seed=SEED, params=self.et_params)
        self.rf = SklearnWrapper(clf=RandomForestClassifier, seed=SEED, params=self.rf_params)
        self.cb = SklearnWrapper(clf=CatBoostClassifier, seed=SEED, params=self.cb_params)
        
    def train(self):
        print("xg..")
        self.xg_oof_train, self.xg_oof_test = get_oof_xgb(self.xg)
        print("et..")
        self.et_oof_train, self.et_oof_test = get_oof(self.et)
        print("rf..")
        self.rf_oof_train, self.rf_oof_test = get_oof(self.rf)
        print("cb..")
        self.cb_oof_train, self.cb_oof_test = get_oof(self.cb)
        
    def after_train(self):
        self.x_train = np.concatenate((self.xg_oof_train, self.et_oof_train, self.rf_oof_train, self.cb_oof_train), axis=1)
        self.x_test = np.concatenate((self.xg_oof_test, self.et_oof_test, self.rf_oof_test, self.cb_oof_test), axis=1)
        np.save('x_train', self.x_train)
        np.save('x_test', self.x_test)
        self.dtrain = xgb.DMatrix(self.x_train, label=self.y_train)
        self.dtest = xgb.DMatrix(self.x_test)
        self.subm = pd.read_csv('submission.csv')
        self.xgb_params = {
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
        self.res = xgb.cv(self.xgb_params, self.dtrain, num_boost_round=500, nfold=4, seed=SEED, stratified=False,
                     early_stopping_rounds=50, verbose_eval=10, show_stdv=True)
        self.best_nrounds = self.res.shape[0] - 1

        print("meta xgb train..")
        self.gbdt = xgb.train(self.xgb_params, self.dtrain, self.best_nrounds)
        self.fi = self.gbdt.predict(self.dtest)
        self.fi = np.array(self.fi)
        np.save('fi', self.fi)

        self.subm['TARGET'] = self.fi
        self.subm.to_csv('Final_result.csv', index=False)

    def get_oof_xgb(self, clf):
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
    
    def get_oof(self, clf):
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
    
    def entire_process(self):
        self.set_train()
        self.train()
        self.after_train()
        

