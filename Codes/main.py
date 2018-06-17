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
from imblearn.over_sampling import SMOTE
from fancyimpute import KNN
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
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
import warnings
warnings.filterwarnings('ignore')
from classifier import *


##Reading from the input data
data_in=pd.read_csv('bank-additional.csv')
data_in=data_in.replace('unknown', np.NaN)
data_in.default=data_in.default.replace(np.NaN,'yes')
#cat_label=cat_values(data_in)
"""
##Plotting the histogram of all the features
for i in cat_label:
    plt.figure(i)
    data_in[i].value_counts().plot(kind='bar')
    plt.title(i)
pd.DataFrame.hist(data_in,grid=False)
##Plotting the correlation between the features
f, ax = plt.subplots(figsize=(10, 8))
corr = data_in.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap='viridis'),
            square=True, ax=ax)
"""



data_in=data_in.dropna(axis=0, how='any')  ##Removing unknown datapoints 
raw_data=data_in.iloc[:,0:19].copy()
data=raw_data.copy()

##splitting the test and training set 
X=data
Y=data_in.iloc[:,19].copy()
X,Y=shuffle(X,Y)
"""
vb=X.describe().columns
pd.plotting.scatter_matrix(X[vb],c=Y.reshape(4119),figsize=(12,12),cmap='flag',alpha=0.2)
"""

##splitting the data into test set and train set
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.4,random_state=0,stratify=Y)


##Data Pre Processing
#======================================

##Taking care of unknown values

##to binarize the class outputs 
lb = preprocessing.LabelBinarizer()
Y_train=lb.fit_transform(Y_train)
Y_test=lb.transform(Y_test)
X_train.contact=lb.fit_transform(X_train.contact)
X_test.contact=lb.transform(X_test.contact)
X_train=X_train.reset_index(drop=True)
X_test=X_test.reset_index(drop=True)
Y_train=pd.DataFrame(Y_train)
Y_test=pd.DataFrame(Y_test)
Y_train=Y_train.reset_index(drop=True)
Y_test=Y_test.reset_index(drop=True)


#X_train = X_train.drop(['euribor3m','age','job','pdays','poutcome','campaign'], 1)
#X_test = X_test.drop(['euribor3m','age','job','pdays','poutcome','campaign'], 1)
Y_train.columns=['label']
Y_test.columns=['label']
cat_label=cat_values(X_train)
"""
#Imputing with SVM
X_train=class_wise_impute(X_train,Y_train)
X_test=class_wise_impute(X_test,Y_test)
"""

##Imputing with mode
CI=CustomImputer(strategy='mode')
CI.fit(X_train[Y_train.label==1])
X_train[Y_train.label==1]=CI.transform(X_train[Y_train.label==1])
CI.fit(X_train[Y_train.label==0])
X_train[Y_train.label==0]=CI.transform(X_train[Y_train.label==0])
CI.fit(X_test)
X_test=CI.transform(X_test)


##Converting 2 categories columns to binary
X_train.housing=lb.fit_transform(X_train.housing)
X_test.housing=lb.transform(X_test.housing)
X_train.loan=lb.fit_transform(X_train.loan)
X_test.loan=lb.transform(X_test.loan)
#X_train = X_train.drop(['month','day_of_week'], 1)
#X_test = X_test.drop(['month','day_of_week'], 1)


##Finding the categorical values and the non categorical values
##Converting categorical values to numerical values
train_length=(len(X_train))
dataset=pd.concat([X_train,X_test],axis=0)
dataset=pd.get_dummies(dataset,columns=cat_label)
X_train=dataset.iloc[0:train_length,:]
X_test=dataset.iloc[train_length:len(dataset),:]


##Standardizing the values
minmax=MinMaxScaler()
X_train=minmax.fit_transform(X_train)
X_test=minmax.transform(X_test)

"""
##Performing PCA to reduce the number of features
pca=PCA(n_components=1,copy=True)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
"""


##Resampling using SMOTE and ENN for increasing datapoints of class 
sm = SMOTE(random_state=100)
X_resampled, Y_resampled = sm.fit_sample(X_train, Y_train)
X_resampled,Y_resampled=shuffle(X_resampled,Y_resampled)

X_test,Y_test=sm.fit_sample(X_test,Y_test)
"""
##choosing the number of features

kbest= SelectKBest(f_classif,k=2)
X_resampled=kbest.fit_transform(X_train, Y_train)
X_test= kbest.transform(X_test)

##Classification Start


X_resampled=X_train.copy()
Y_resampled=Y_train.copy()

kbest= SelectKBest(f_classif,k=50)
X_resampled=kbest.fit_transform(X_train, Y_train)
X_resampled_test= kbest.transform(X_test)
"""
"""
for i in range(1,2):
    
    

"""
"""
pca=PCA(n_components=50,copy=True )
X_resampled=pca.fit_transform(X_train)
X_resampled_test=pca.transform(X_test)
"""
classify(X_resampled, Y_resampled, X_test, Y_test)

    

"""
##Cross Validation Code for SVM
X_resampled=X_train.copy()
Y_resampled=Y_train.copy()

mean=[]
var=[]
C=np.logspace(-3,3,num=10) 
G=np.logspace(-3,3,num=10)
ACC=np.zeros((30,1))

from sklearn.model_selection import StratifiedKFold
skf=StratifiedKFold(n_splits=3,shuffle=True)
Y_resampled=pd.DataFrame.as_matrix(Y_resampled)


for i in range(10):
    for j in range(10):
        c=C[i]
        g=G[j]
        accuracy=[]
        for train_index, test_index in skf.split(X_resampled,Y_resampled):
            X_cv_train, X_cv_test=X_resampled[train_index],X_resampled[test_index]
            Y_cv_train, Y_cv_test=Y_resampled[train_index],Y_resampled[test_index]
            clf=SVC(probability=True,kernel='rbf',class_weight='balanced',gamma=g, C=c)
            clf.fit(X_cv_train,Y_cv_train)
            y_pred=clf.predict(X_cv_test)
            acc=f1_score(Y_cv_test,y_pred,average='weighted')
            accuracy.append(acc)
        
        mean.append(np.mean(accuracy))
        var.append(np.var(accuracy))
        ACC[i,j]=np.mean(accuracy)
        print("Iteration number: {}".format(i*50+j))

index=np.argmax(mean)
best_C=C[index//10]
best_g=G[index%10]
svc_final=SVC(C=best_C, kernel='rbf',gamma=best_g,class_weight='balanced')
svc_final.fit(X_resampled,Y_resampled)
y_pred=svc_final.predict(X_train)
print("After Cross Validation for SVM RBF Kernel")
acc=f1_score(Y_train,y_pred,average='weighted')
error_2=roc_auc_score(Y_train,y_pred,average='weighted')
print("Linear SVM Test f1_accuracy: {}, {}".format(acc,error_2))
acc_degree.append(acc)
"""




