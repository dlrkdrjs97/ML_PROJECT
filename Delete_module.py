
# coding: utf-8


## DELETE MODULE



## IMPORT EXTERN MODULE
import pandas as pd
import numpy as np



class delete:
    def setdata(self, merged_df, train_df, importance):
        self.merged_df = merged_df
        self.train_df = train_df
        self.importance = importance
        
    def delete_process(self):
        merge = self.merged_df
        for i in range(len(self.train_df.columns)):
            if self.importance[i] == 0 :
                merge = merge.drop(self.train_df.columns[i],axis =1)
                # Store data
                merge.to_csv('newmerge.csv', index=False)
        return merge
    def entire_process(self):
        merged_df = self.delete_process()
        return merged_df
                       

