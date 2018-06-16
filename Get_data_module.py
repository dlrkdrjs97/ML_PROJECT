
# coding: utf-8


## GET_DATA_MODULE



## IMPORT EXTERN MODULE
import pandas as pd
import os


sample_size = None

class getdata:
    def get_dir(self, dir):
        self.dir = dir
    def get_apptrain(self):
        app_train = pd.read_csv(os.path.join(self.dir, 'application_train.csv'), nrows=sample_size)
        return app_train
    def get_apptest(self):
        app_test = pd.read_csv(os.path.join(self.dir, 'application_test.csv'), nrows=sample_size)
        return app_test
    def get_bureau(self):
        bureau = pd.read_csv(os.path.join(self.dir, 'bureau.csv'), nrows=sample_size)
        return bureau
    def get_bureaubal(self):
        bureaubal = pd.read_csv(os.path.join(self.dir, 'bureau_balance.csv'), nrows=sample_size)
        return bureaubal
    def get_credit(self):
        credit = pd.read_csv(os.path.join(self.dir, 'credit_card_balance.csv'), nrows=sample_size)
        return credit
    def get_pos(self):
        pos = pd.read_csv(os.path.join(self.dir, 'POS_CASH_balance.csv'), nrows=sample_size)
        return pos
    def get_prev(self):
        prev = pd.read_csv(os.path.join(self.dir, 'previous_application.csv'), nrows=sample_size)
        return prev
    def get_install(self):
        install = pd.read_csv(os.path.join(self.dir, 'installments_payments.csv'), nrows=sample_size)
        return install
        

