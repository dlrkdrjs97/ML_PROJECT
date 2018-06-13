from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
np.set_printoptions(threshold=np.nan)
import numpy as np
import pickle

class trainning_prediction:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.test_size = 0.2
        self.random_state = 42 
    
    def entire_training_process(self):
        print("----------ENTIRE_TRAINING_PROCESS----------")
        self.extracting_features()
        self.preprocessing_data()
        self.training_data()
        self.save_model()

    def entire_testing_process(self):
        print("----------ENTIRE_TESTING_PROCESS----------")
        self.retrieve_model()
        #self.predicting_data()

    def extracting_features(self):
        print("ENTIRE_TRAINING_PROCESS - extracting_features")
        self.dtypes = self.train_data.dtypes
        self.dtypes = self.dtypes[self.dtypes != object]
        self.features = list(set(self.dtypes.index)-set(['TARGET'])-set(['SK_ID_CURR']))
        with open('features.txt', 'wb') as fp:
            pickle.dump(self.features, fp)

    def preprocessing_data(self):
        self.X = self.train_data[self.features]
        self.Y = self.train_data['TARGET']
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y, test_size=self.test_size, random_state=self.random_state)

        
    def training_data(self):
        print("ENTIRE_TRAINING_PROCESS - training_data")
        self.clf1 = LogisticRegression(random_state = 1)
        self.clf2 = RandomForestClassifier(random_state = 1)
        self.clf3 = GaussianNB()

        self.eclf  = VotingClassifier(estimators = [('lr', self.clf1), ('rf', self.clf2), ('gnb', self.clf3)], voting = 'hard')
        for clf , label in zip([self.clf1, self.clf2, self.clf3, self.eclf],['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
            scores = cross_val_score(clf, self.X_train, self.Y_train, cv=5, scoring='accuracy')
            print("Accuracy: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))

    def save_model(self):
        print("ENTIRE_TRAINING_PROCESS - save_model")
        pickle.dump(self.clf1, open("clf1.dat", "wb"))
        pickle.dump(self.clf2, open("clf2.dat", "wb"))
        pickle.dump(self.clf3, open("clf3.dat", "wb"))
        pickle.dump(self.eclf, open("eclf.dat", "wb"))

    def retrieve_model(self):
        self.clf1 = pickle.load(open("clf1.dat", "rb"))
        self.clf2 = pickle.load(open("clf2.dat", "rb"))
        self.clf3 = pickle.load(open("clf3.dat", "rb"))
        self.eclf = pickle.load(open("eclf.dat", "rb"))

    #def predicting_data(self, test_data):
