# EXTERN MODULE IMPORT
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


## MODULE IMPORT
import MODULE1 

## READ DATA SET
df = pd.read_csv("application_train.csv")
df_test = pd.read_csv("application_test.csv")

## MODULE # 1
m1 = MODULE1.XGB()
m1.read_data(df, df_test)
m1.entire_process()
acc_train, acc_test = m1.figure_out_training_data()
print("Train AUC %.4f"%acc_train)
print("Test AUC %.4f"%acc_test)

m1._prediction()