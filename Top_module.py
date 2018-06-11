# EXTERN MODULE IMPORT
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np


## MODULE IMPORT
import MODULE1 
import MODULE2
import MODULE3
import MODULE4
import MODULE5
import PREDICTION

## READ DATA SET
df = pd.read_csv("application_train.csv")
df_test = pd.read_csv("application_test.csv")

## MODULE # 1
m1 = MODULE1.XGB()
m1.read_data(df)
ret_val = m1.entire_training_process()
acc_train, acc_test = m1.figure_out_training_data()
po1 = m1.predict_out()
print(type(po1))
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 1
m2 = MODULE2.XGB()
m2.read_data(df)
ret_val = m2.entire_training_process()
acc_train, acc_test = m2.figure_out_training_data()
po2 = m2.predict_out()
print(type(po2))
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 1
m3 = MODULE3.XGB()
m3.read_data(df)
ret_val = m3.entire_training_process()
acc_train, acc_test = m3.figure_out_training_data()
po3 = m3.predict_out()
print(type(po3))
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 1
m4 = MODULE1.XGB()
m4.read_data(df)
ret_val = m4.entire_training_process()
acc_train, acc_test = m4.figure_out_training_data()
po4 = m4.predict_out()
print(type(po4))
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 1
m5 = MODULE1.XGB()
m5.read_data(df)
ret_val = m5.entire_training_process()
acc_train, acc_test = m5.figure_out_training_data()
po5 = m5.predict_out()
print(type(po5))
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)


label = df['TARGET']
label = np.array(list(label))
'''
po1 = po1.reshape(-1,1)
po2 = po2.reshape(-1,1)
po3 = po3.reshape(-1,1)
po4 = po4.reshape(-1,1)
po5 = po5.reshape(-1,1)
label = label.reshape(-1,1)
'''
IN = np.column_stack((po1, po2, po3, po4, po5))
print(IN[:10])
pd = PREDICTION.PREDICT(IN, label)
pd.train()