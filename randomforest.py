import numpy as np
import pandas as pd
from sklearn import svm,metrics,cross_validation
from sklearn.ensemble import RandomForestClassifier

#Read training data and split into train and test data
data=pd.read_csv('train.csv')
data1=data.values
X=data1[:,1:]
y=np.ravel(y)
Xtrain,Xtest,ytrain,ytest=cross_validation.train_test_split(X,y,test_size=0.25)

#Run RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(Xtrain,ytrain)
ypred=rf.predict(Xtest) 
metrics.precision_score(ytest,ypred)

