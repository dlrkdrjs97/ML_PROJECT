
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



## LOAD DATA
from Get_data_module import getdata

input_dir = os.path.join(os.pardir, 'ML_PROJECT')
gd = getdata(input_dir)
app_train_df = gd.get_apptrain()
app_test_df = gd.get_apptest()
bureau_df = gd.get_bureau()
bureau_balance_df = gd.get_bureaubal()
credit_card_df = gd.get_credit()
pos_cash_df = gd.get_pos()
prev_app_df = gd.get_prev()
install_df = gd.get_install()
# for cheking
print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
print('Main application test data set shape = {}'.format(app_test_df.shape))
print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))



## PREPROCESS DATA

from Preprocessing_module import preprocess

preprocess.setdata(app_train_df, app_test_df, bureau_df, bureau_balance_df, credit_card_df, pos_cash_df, prev_app_df, install_df)
train_df, test_df, categorical_feats, len_train, meta_df, merged_df = preprocess.entire_process()




## FEATURE EXTRACTION
from Feature_extraction_module import extraction
extraction.setdata(train_df, test_df, categorical_feats, len_train, meta_df)
importance = extraction.entire_process()




## DELETE FEATURE
from Delete_module import delete
delete.setdata(merged_df, train_df, importance)
merged_df = delete.entire_process()



## ENSEMBLE TRAINING
from Ensemble_module import ensembletrain
ensemble.setdata(merged_df, len_train)
ensemble.entire_process()

