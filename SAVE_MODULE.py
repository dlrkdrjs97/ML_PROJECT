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

class saveresult:
    def __init__(self, test_data, test_result):
        self.test_data = test_data
        self.test_result = test_result

    def entire_process(self):
        self.open_features()
        self.store_data()

    def open_features(self):
        print("I AM PREPROCESSING DATA !")
        with open ('features_module1.txt', 'rb') as fp:
            self.features = pickle.load(fp)
        print("DONE !")

    def store_data(self):
        self.Y_test = self.test_data[self.features]
        self.results = self.test_data[["SK_ID_CURR"]]
        self.results["TARGET"] = self.test_result
        self.results.to_csv("results.csv",index=False,columns=self.results.columns)
