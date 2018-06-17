import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from categorical_imputer import CustomImputer
from categorical_imputer import Imputing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN
from fancyimpute import KNN
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from impute import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve,auc
import warnings
warnings.filterwarnings('ignore')



def datafill(X):
    uk_col=X.isna().any().tolist
    X=pd.DataFrame(X)
    #Y=pd.DataFrame(Y)
    X_train=pd.DataFrame
    X_multiclass=pd.DataFrame
    X_multiclass=X.loc[:, X.isna().any()]
    X_train = X.loc[:,~X.isna().any()]
    for i in range(X_multiclass.shape[1]):
        current=pd.DataFrame(X_multiclass.iloc[:,i])
        label=fitting(X_train,current)
        X_train=pd.concat([X_train,label],axis=1)
    return X_train
    
    
    
def fitting(X_train,X_multiclass):
    Data=pd.DataFrame.empty
    Data=pd.concat([X_train,X_multiclass],axis=1)
    Data_train=Data.dropna(axis=0, how='any')
    Data_test= Data[Data.iloc[:,X_train.shape[1]].isnull()]
    train = Data_train.iloc[:,0:Data_train.shape[1]-1]
    test = Data_test.iloc[:,0:Data_test.shape[1]-1]
    train_label=Data_train.iloc[:,-1]  
    train = train.drop(['month','day_of_week'], 1)
    test = test.drop(['month','day_of_week'], 1)
    train=pd.get_dummies(train)
    test=pd.get_dummies(test)
    l1=list(train)
    l2=list(test)
    intr1=list(set(l1).symmetric_difference(l2))
    train.drop(intr1, axis=1, inplace=True)
    test_label=classify(train,train_label,test)
    test_label=pd.DataFrame(test_label)
    test_label.index=test.index
    label=pd.concat([test_label,train_label])
    name=[X_multiclass.columns[0]]
    label.columns=name
    return label

def classify(X_train,Y_train,X_test):
    #svc=SVC(kernel='rbf',gamma=100)
    rf=SVC()
    rf.fit(X_train,Y_train)
    Y_test=rf.predict(X_test)
    return Y_test


def class_wise_impute(X,Y):
    X=pd.concat([X,Y],axis=1)
    X_class1=X.loc[X['label']==1].iloc[:,0:X.shape[1]-1]
    class1=datafill(X_class1)
    X_class0=X.loc[X['label']==0].iloc[:,0:X.shape[1]-1]
    class0=datafill(X_class0)
    return_data=pd.concat([class1,class0],axis=0)
    return return_data



##Finding the categorical values and the non categorical values
def cat_values(X_train):
    datatypes=X_train.dtypes
    cat_values=[]
    cat_label=[]
    for i in range(X_train.shape[1]):
        if(datatypes[i]=='object'):
            cat_values.append(i) ##columns having categorical values
    for i in range(len(cat_values)):
        cat_label.append(X_train.columns[cat_values[i]])
    return cat_label

def roc_auc_plot(Y_test,Y_pred,prob,name):
            fpr, tpr, thresholds = roc_curve(Y_test, prob[:,1])
            r_a = auc(fpr, tpr)
            roc_auc = roc_auc_score(Y_test,Y_pred)
            #Using auc_score
            print("ROC_Curve:\t",r_a)
            #print("roc_auc:\n",roc_auc)
            lw = 2
            #print("AUC:\n",au,"\n")
            plt.figure()
            plt.plot(fpr, tpr,lw=lw, color = 'red')
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic: '+str(name))
            plt.legend()
            plt.show()
            print("Done")