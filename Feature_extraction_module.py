

## FEATURE EXTRACTION MODULE


# In[1]:


## IMPORT EXTERN MODULE
import pandas as pd
import numpy as np
import lightgbm as lgbm




class extraction:
    def setdata(self, train_df, test_df, categorical_feats, len_train, meta_df):
        self.train_df = train_df
        self.test_df = test_df
        self.target = self.train_df.pop('TARGET')
        self.test_df.drop(columns='TARGET', inplace=True)
        self.categorical_feats = categorical_feats
        self.len_train = len_train
        self.meta_df = meta_df
        
    def lgbmtrain(self):
        self.lgbm_train = lgbm.Dataset(data=self.train_df,
                          label=self.target,
                          categorical_feature=self.categorical_feats,
                          free_raw_data=False)
        self.lgbm_params = {
            'boosting': 'dart',
            'application': 'binary',
            'learning_rate': 0.1,
            'min_data_in_leaf': 30,
            'num_leaves': 31,
            'max_depth': -1,
            'feature_fraction': 0.5,
            'scale_pos_weight': 2,
            'drop_rate': 0.02
        }
        self.cv_results = lgbm.cv(train_set=self.lgbm_train,
                     params=self.lgbm_params,
                     nfold=4,
                     num_boost_round=2000,
                     early_stopping_rounds=50,
                     stratified=True,
                     verbose_eval=50,
                     metrics=['auc'])
        self.optimum_boost_rounds = np.argmax(self.cv_results['auc-mean'])
        print('Optimum boost rounds = {}'.format(self.optimum_boost_rounds))
        print('Best LGBM CV result = {}'.format(np.max(self.cv_results['auc-mean'])))
        self.clf = lgbm.train(train_set=self.lgbm_train,
                 params=self.lgbm_params,
                 num_boost_round=self.optimum_boost_rounds)
        
    def lgbmpredict(self):
        self.y_pred = self.clf.predict(self.test_df)
        self.out_df = pd.DataFrame({'SK_ID_CURR': self.meta_df['SK_ID_CURR'][self.len_train:], 'TARGET': self.y_pred})
        # Store data
        self.out_df.to_csv('feature_prediction.csv', index=False)
    
    def plot_importance(self):
        fig, (ax, ax1) = plt.subplots(1, 2, figsize=[11, 7])
        lgbm.plot_importance(self.clf, ax=ax, max_num_features=50, importance_type='split')
        lgbm.plot_importance(self.clf, ax=ax1, max_num_features=50, importance_type='gain')
        ax.set_title('Importance by splits')
        ax1.set_title('Importance by gain')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
    def get_importance(self):
        importance = lgbm.Booster.feature_importance(self.clf,importance_type = 'split')
        return importance
        
    def entire_process(self):
        self.lgbmtrain()
        self.lgbmpredict()
        self.plot_importance()
        self.importance = self.get_importance()
        return self.importance
        

