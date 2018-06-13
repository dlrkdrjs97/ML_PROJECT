## Import extern module
import pandas as pd
import prediction_module as pm

## Read data set
train_data = pd.read_csv("application_train.csv")
test_data = pd.read_csv("application_test.csv")

## Training Session
tp = pm.trainning_prediction(train_data, test_data)
tp.entire_training_process()