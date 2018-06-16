
## PREPROCESSING MODULE ##



## IMPORT EXTERN MODULE
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class preprocess:

    def __init__(self, ap_tr, ap_te, br, br_bal, credit, pos, prev, install):
        self.app_train_df = ap_tr
        self.app_test_df = ap_te
        self.bureau_df = br
        self.bureau_balance_df = br_bal
        self.credit_card_df = credit
        self.pos_cash_df = pos
        self.prev_app_df = prev
        self.install_df = install
    
    def agg_and_merge(self, left_df, right_df, agg_method, right_suffix):
        """ Aggregate a df by 'SK_ID_CURR' and merge it onto another.
        This method allows feature name """

        agg_df = right_df.groupby('SK_ID_CURR').agg(agg_method)
        merged_df = left_df.merge(agg_df, left_on='SK_ID_CURR', right_index=True, how='left',
                                  suffixes=['', '_' + right_suffix + agg_method.upper()])
        return merged_df
    
    def process_dataframe(self, input_df, encoder_dict= None):
        """ Process a dataframe into a form useable by LightGBM """

        # Label encode categoricals
        print('Label encoding categorical features...')
        categorical_feats = input_df.columns[input_df.dtypes == 'object']



        for feat in categorical_feats:
            encoder = LabelEncoder()
            input_df[feat] = encoder.fit_transform((input_df[feat].fillna('NULL')).astype(str))
            print(input_df[feat])
        print('Label encoding complete.')

        return input_df, categorical_feats.tolist(), encoder_dict
    
    def feature_engineering(self, app_data, bureau_df, bureau_balance_df, credit_card_df,
                        pos_cash_df, prev_app_df, install_df):
        """ 
        Process the input dataframes into a single one containing all the features. Requires
        a lot of aggregating of the supplementary datasets such that they have an entry per
        customer.

        Also, add any new features created from the existing ones
        """

        # # Add new features

        # Amount loaned relative to salary
        app_data['LOAN_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
        app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']

        # Number of overall payments (I think!)
        app_data['ANNUITY LENGTH'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']

        # Social features
        app_data['WORKING_LIFE_RATIO'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
        app_data['INCOME_PER_FAM'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
        app_data['CHILDREN_RATIO'] = app_data['CNT_CHILDREN'] / app_data['CNT_FAM_MEMBERS']

        # A lot of the continuous days variables have integers as missing value indicators.
        prev_app_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
        prev_app_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
        prev_app_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
        prev_app_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
        prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)

        # # Aggregate and merge supplementary datasets

        # Previous applications
        print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
        agg_funs = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}
        prev_apps = prev_app_df.groupby('SK_ID_CURR').agg(agg_funs)
        prev_apps.columns = ['PREV APP COUNT', 'TOTAL PREV LOAN AMT']
        merged_df = app_data.merge(prev_apps, left_on='SK_ID_CURR', right_index=True, how='left')

        # Average the rest of the previous app data
        for agg_method in ['mean', 'max', 'min']:
            merged_df = self.agg_and_merge(merged_df, prev_app_df, agg_method, 'PRV')
        print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

        # Previous app categorical features
        prev_app_df, cat_feats, _ = self.process_dataframe(prev_app_df)
        prev_apps_cat_avg = prev_app_df[cat_feats + ['SK_ID_CURR']].groupby('SK_ID_CURR')                                 .agg({k: lambda x: str(x.mode().iloc[0]) for k in cat_feats})
        merged_df = merged_df.merge(prev_apps_cat_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_BAVG'])
        print('Shape after merging with previous apps cat data = {}'.format(merged_df.shape))

        # Credit card data - numerical features
        wm = lambda x: np.average(x, weights=-1/credit_card_df.loc[x.index, 'MONTHS_BALANCE'])
        credit_card_avgs = credit_card_df.groupby('SK_ID_CURR').agg(wm)   
        merged_df = merged_df.merge(credit_card_avgs, left_on='SK_ID_CURR', right_index=True,
                                    how='left', suffixes=['', '_CC_WAVG'])
        for agg_method in ['mean', 'max', 'min']:
            merged_df = self.agg_and_merge(merged_df, credit_card_avgs, agg_method, 'CC')
        print('Shape after merging with previous apps num data = {}'.format(merged_df.shape))

        # Credit card data - categorical features
        most_recent_index = credit_card_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
        cat_feats = credit_card_df.columns[credit_card_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
        merged_df = merged_df.merge(credit_card_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                           how='left', suffixes=['', '_CCAVG'])
        print('Shape after merging with credit card data = {}'.format(merged_df.shape))

        # Credit bureau data - numerical features
        for agg_method in ['mean', 'max', 'min']:
            merged_df = self.agg_and_merge(merged_df, bureau_df, agg_method, 'B')
        print('Shape after merging with credit bureau data = {}'.format(merged_df.shape))

        # Bureau balance data
        most_recent_index = bureau_balance_df.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
        bureau_balance_df = bureau_balance_df.loc[most_recent_index, :]
        merged_df = merged_df.merge(bureau_balance_df, left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU',
                                how='left', suffixes=['', '_B_B'])
        print('Shape after merging with bureau balance data = {}'.format(merged_df.shape))

        # Pos cash data - weight values by recency when averaging
        wm = lambda x: np.average(x, weights=-1/pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])
        f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF':wm}
        cash_avg = pos_cash_df.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                                                     'SK_DPD', 'SK_DPD_DEF'].agg(f)
        merged_df = merged_df.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                    how='left', suffixes=['', '_CAVG'])

        # Unweighted aggregations of numeric features
        for agg_method in ['mean', 'max', 'min']:
            merged_df = self.agg_and_merge(merged_df, pos_cash_df, agg_method, 'PC')

        # Pos cash data data - categorical features
        most_recent_index = pos_cash_df.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
        cat_feats = pos_cash_df.columns[pos_cash_df.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
        merged_df = merged_df.merge(pos_cash_df.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', right_on='SK_ID_CURR',
                           how='left', suffixes=['', '_CAVG'])
        print('Shape after merging with pos cash data = {}'.format(merged_df.shape))

        # Installments data
        for agg_method in ['mean', 'max', 'min']:
            merged_df = self.agg_and_merge(merged_df, install_df, agg_method, 'I')    
        print('Shape after merging with installments data = {}'.format(merged_df.shape))

        # Add more value counts
        merged_df = merged_df.merge(pd.DataFrame(bureau_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                    right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])
        merged_df = merged_df.merge(pd.DataFrame(credit_card_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                    right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])
        merged_df = merged_df.merge(pd.DataFrame(pos_cash_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                    right_index=True, how='left', suffixes=['', '_CNT_POS_CASH'])
        merged_df = merged_df.merge(pd.DataFrame(install_df['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                    right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
        print('Shape after merging with counts data = {}'.format(merged_df.shape))

        return merged_df

    def entire_process(self):
        self.len_train = len(self.app_train_df)
        self.app_both = pd.concat([self.app_train_df, self.app_test_df])
        self.merged_df = self.feature_engineering(self.app_both, self.bureau_df, self.bureau_balance_df, self.credit_card_df,
                                self.pos_cash_df, self.prev_app_df, self.install_df)
        # Store data
        self.merged_df.to_csv('feature_data.csv', index=False)
        # Separate metadata
        self.meta_cols = ['SK_ID_CURR']
        self.meta_df = self.merged_df[self.meta_cols]
        self.merged_df.drop(columns=self.meta_cols, inplace=True)
        # Process the data set.
        self.merged_df, self.categorical_feats, self.encoder_dict = self.process_dataframe(input_df = self.merged_df)
        # Re-separate into train and test
        self.train_df = self.merged_df[:self.len_train]
        self.test_df = self.merged_df[self.len_train:]
        
        self.non_obj_categoricals = [
            'FONDKAPREMONT_MODE', 'HOUR_APPR_PROCESS_START', 'HOUSETYPE_MODE',
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
            'NAME_INCOME_TYPE', 'NAME_TYPE_SUITE', 'OCCUPATION_TYPE',
            'ORGANIZATION_TYPE', 'STATUS', 'NAME_CONTRACT_STATUS_CAVG',
            'WALLSMATERIAL_MODE', 'WEEKDAY_APPR_PROCESS_START', 'NAME_CONTRACT_TYPE_BAVG',
            'WEEKDAY_APPR_PROCESS_START_BAVG', 'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 
            'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON', 'NAME_TYPE_SUITE_BAVG', 
            'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 
            'NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 
            'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION', 'NAME_CONTRACT_STATUS_CCAVG',
            'CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE'
        ]
        self.categorical_feats = self.categorical_feats + self.non_obj_categoricals
        
        return (self.train_df, self.test_df, self.categorical_feats, self.len_train, self.meta_df, self.merged_df)

