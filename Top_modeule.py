# EXTERN MODULE IMPORT
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


## MODULE IMPORT
import MODULE1 
import MODULE2
import MODULE3
import MODULE4
import MODULE5

## READ DATA SET
df = pd.read_csv("application_train.csv")
df_test = pd.read_csv("application_test.csv")

## MODULE # 1
print("MODULE # 1")
m1 = MODULE1.XGB()
m1.read_data(df)
ret_val = m1.entire_training_process()
acc_train, acc_test = m1.figure_out_training_data()
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 2
print("MODULE # 2")
m2 = MODULE2.XGB()
m2.read_data(df)
ret_val = m2.entire_training_process()
acc_train, acc_test = m2.figure_out_training_data()
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 3
print("MODULE # 3")
m3 = MODULE3.XGB()
m3.read_data(df)
ret_val = m3.entire_training_process()
acc_train, acc_test = m3.figure_out_training_data()
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 4
print("MODULE # 4")
m4 = MODULE1.XGB()
m4.read_data(df)
ret_val = m4.entire_training_process()
acc_train, acc_test = m4.figure_out_training_data()
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

## MODULE # 5
print("MODULE # 5")
m5 = MODULE1.XGB()
m5.read_data(df)
ret_val = m5.entire_training_process()
acc_train, acc_test = m5.figure_out_training_data()
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)
