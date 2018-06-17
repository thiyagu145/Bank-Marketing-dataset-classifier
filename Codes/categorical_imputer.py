import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mean',filler='NA'):
       self.strategy = strategy
       self.fill = filler

    def fit(self, X, y=None):
       if self.strategy in ['mean','median']:
           if not all(X.dtypes == np.number):
               raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
       if self.strategy == 'mean':
           self.fill = X.mean()
       elif self.strategy == 'median':
           self.fill = X.median()
       elif self.strategy == 'mode':
           self.fill = X.mode().iloc[0]
       elif self.strategy == 'fill':
           if type(self.fill) is list and type(X) is pd.DataFrame:
               self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
       return self

    def transform(self, X, y=None):
       return X.fillna(self.fill)
   
    
class Imputing():
    
    def __init__(self,X_train,Y_train,clsfr):
        self.X_train=X_train
        self.Y_train=Y_train
        self.clsfr=clsfr
    

