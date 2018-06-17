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
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from impute import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

def classify(X_train, Y_train, X_test, Y_test):
    print()
    print()
    print("Starting to train the classification models")
    print()
    print()
    """
    print("Perceptron Model")
    print()
    ##Perceptron Training Model
    percep = Perceptron(max_iter=10000,warm_start=False,class_weight='balanced')
    percep.fit(X_train,Y_train)
    Pred_train=percep.predict(X_train)
    Pred_test=percep.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("Perceptron Model Training Error: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("Perceptron Model Test Error: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    #roc_auc_plot(Y_test,Pred_test,percep.predict_proba(X_test),'Perceptron')
    
    ##SVM Linear Training Model
    clf=SVC(C=13.73,kernel='linear',probability=True,class_weight='balanced')
    clf.fit(X_train,Y_train)
    Pred_train=clf.predict(X_train)
    Pred_test=clf.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("SVM Linear Model Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("SVM Linear Model Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    roc_auc_plot(Y_test,Pred_test,clf.predict_proba(X_test),'SVM Linear')
    print()
    print()
    """
    ##SVM RBF Training Model
    print("SVM RBF Kernel Model")
    print()
    clf=SVC(probability=True,kernel='rbf',class_weight='balanced',gamma=0.004641, C=100)
    clf.fit(X_train,Y_train)
    Pred_train=clf.predict(X_train)
    Pred_test=clf.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("SVM RBF kernel Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("SVM RBF Kernel Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    print()
    print()
    roc_auc_plot(Y_test,Pred_test,clf.predict_proba(X_test),'SVM RBF')
    """
    ##SVM Poly Training Model
    print("SVM Poly Kernel Model")
    print()
    clf=SVC(probability=True,kernel='poly',degree=5,gamma=4.4, C=210,class_weight='balanced')
    clf.fit(X_train,Y_train)
    Pred_train=clf.predict(X_train)
    Pred_test=clf.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("SVM Poly Kernel Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("SVM Poly Kernel Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    roc_auc_plot(Y_test,Pred_test,clf.predict_proba(X_test),'SVM Poly')
    print()
    print()
    
 
    
    ##Nearest Neighbors
    knn=KNeighborsClassifier(n_neighbors=6,p=1,weights='uniform',algorithm='auto')
    knn.fit(X_train,Y_train)
    Pred_train=knn.predict(X_train)
    Pred_test=knn.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("KNN Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("KNN Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    roc_auc_plot(Y_test,Pred_test,knn.predict_proba(X_test),'KNN')
 

    ##Random Forest
    rfc=RandomForestClassifier(n_estimators=14,bootstrap=True,class_weight='balanced')
    rfc.fit(X_train,Y_train)
    Pred_train=rfc.predict(X_train)
    Pred_test=rfc.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("Random Forest Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("Random Forest Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    roc_auc_plot(Y_test,Pred_test,rfc.predict_proba(X_test),'Random Forest')

    
    ##Naive Bayes classifier
    naiveB=GaussianNB()
    naiveB.fit(X_train,Y_train)
    Pred_train=naiveB.predict(X_train)
    Pred_test=naiveB.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("Naive Bayes Model Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("Naive Bayes Model Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    roc_auc_plot(Y_test,Pred_test,naiveB.predict_proba(X_test),'Naive Bayes')
    
    ##MLP Classifier
    gbc=MLPClassifier(hidden_layer_sizes=(160,))
    gbc.fit(X_train,Y_train)
    Pred_train=gbc.predict(X_train)
    Pred_test=gbc.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("MLP Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("MLP Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    roc_auc_plot(Y_test,Pred_test,gbc.predict_proba(X_test),'MLP')

    
    ##MLP Classifier
    gbc=DecisionTreeClassifier()
    gbc.fit(X_train,Y_train)
    Pred_train=gbc.predict(X_train)
    Pred_test=gbc.predict(X_test)
    print("Classification model report for Training data")
    train_report=classification_report(Y_train,Pred_train)
    print(train_report)
    print("Classification model report for Test data")
    test_report=classification_report(Y_test,Pred_test)
    print(test_report)
    train_error=f1_score(Y_train,Pred_train,average='weighted')
    train_error_2=roc_auc_score(Y_train,Pred_train,average='weighted')
    test_error=f1_score(Y_test,Pred_test,average='weighted')
    test_error_2=roc_auc_score(Y_test,Pred_test,average='weighted')
    print("Decision Tree Training Performance: f1={}, ROC_AUC={}".format(train_error,train_error_2))
    print("Decision Test Performance: f1={}, ROC_AUC={}".format(test_error,test_error_2))
    roc_auc_plot(Y_test,Pred_test,gbc.predict_proba(X_test),'Decision tree')
    """


