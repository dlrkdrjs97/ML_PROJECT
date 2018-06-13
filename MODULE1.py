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
np.set_printoptions(threshold=np.nan)

class XGB :
    def __init__(self):
        self.test_size = 0.2
        self.random_state = 42
        self.max_depth = 6
        self.learning_rate = 0.1
        self.n_estimators = 100
        self.n_jobs = 16
        self.scale_pos_weight = 4
        self.missing = np.nan
        self.gamma = 16
        self.eval_metric = "auc"
        self.reg_lambda = 40
        self.reg_alpha = 40

    def entire_training_process(self):
        self.preprocess_data_for_trainset()
        self.get_train_data()
        self.training_data()
        self.save_module()
        

    def entire_test_process(self, test_data):
        self.test_data = test_data
        self.preprocess_data_for_testset()
        self.retrieve_module()
        ret_val = self._prediction()
        return ret_val

    def read_data(self, train_data):
        print("I AM READING DATA !")
        self.train_data = train_data
        print("DONE !")

    def preprocess_data_for_testset(self):
        print("I AM PREPROCESSING DATA !")
        with open ('features_module1.txt', 'rb') as fp:
            self.features = pickle.load(fp)
        #print(type(self.features))
        print("DONE !") 

    def preprocess_data_for_trainset(self):
        print("I AM PREPROCESSING DATA !")
        self.dtypes = self.train_data.dtypes
        self.dtypes = self.dtypes[self.dtypes!='object']
        self.features = list(set(self.dtypes.index)-set(['TARGET']))
        #print(self.features)
        with open('features_module1.txt', 'wb') as fp:
            pickle.dump(self.features, fp)
        print("DONE !")
        # Do nothing here

    def get_train_data(self):
        print("I AM GETTING TRAIN DATA !")
        self.X = self.train_data[self.features]
        self.Y = self.train_data['TARGET']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, test_size=self.test_size, random_state=self.random_state)
        print("DONE !")

    def training_data(self):
        print("I AM TRAINING DATA !")
        self.model = XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate, n_estimators=self.n_estimators, n_jobs=self.n_jobs, scale_pos_weight=self.scale_pos_weight ,missing=self.missing ,gamma=self.gamma, eval_metric=self.eval_metric ,reg_lambda=self.reg_lambda,reg_alpha=self.reg_alpha)
        self.model.fit(self.X_train,self.y_train)
        print("DONE !")
        
    def return_model(self):
        return self.model

    def predict_out(self):
        return self.model.predict_proba(self.X)[:,1]
    def save_module(self):
        print("I Am SAVING MODEL !")
        pickle.dump(self.model, open("module1.dat", "wb"))
        print("DONE !")

    def retrieve_module(self):
        print("I Am RETRIEVING MODEL !")
        self.model = pickle.load(open("module1.dat", "rb"))
        print("DONE !")

    def figure_out_training_data(self):
        print("I Am FIGURING OUT DATA !")
        self.X_train_prediction = self.model.predict_proba(self.X_train)[:,1]
        self.X_test_prediction = self.model.predict_proba(self.X_test)[:,1]
        print("DONE !")
        return (roc_auc_score(self.y_train, self.X_train_prediction),roc_auc_score(self.y_test, self.X_test_prediction))
    
    def _prediction(self):
        print("I AM PREDICTING DATA !")
        '''
        self.Y_test = self.test_data[self.features]
        self.results = self.test_data[["SK_ID_CURR"]]
        self.results["TARGET"] = self.model.predict_proba(self.Y_test)[:,1]
        self.results.to_csv("results_from_modeul1.csv",index=False,columns=self.results.columns)
        '''
        self.Y_test = self.test_data[self.features]
        return self.model.predict_proba(self.Y_test)[:,1]
        print("DONE !")
        
   