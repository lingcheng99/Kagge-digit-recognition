import numpy as np
import pandas as pd
from sklearn import metrics,cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import svm

#Read training data and split into train and test data
data=pd.read_csv('train.csv')
data1=data.values
X=data1[:,1:]
y=data1[:,:1]
y=np.ravel(y)
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25)

#Run linear kernel first
svmL1=svm.SVC(kernel='linear',C=0.01)
svmL1.fit(Xtrain,ytrain) 
predL1=svmL1.predict(Xtest) 
print("Classification report for classifier %s:\n%s\n"
      % (svmL1, metrics.classification_report(ytest,predL1)))

#Run gaussian kernel but the result is poor and running time is long
svmR1=svm.SVC(kernel='rbf',gamma=0.001, C=10000)
svmR1.fit(Xtrain,ytrain)
predR1=svmR1.predict(Xtest)
print("Classification report for classifier %s:\n%s\n"
      % (svmR1, metrics.classification_report(ytest,predR1)))

#Run polynomial kernel
svmP1=svm.SVC(kernel='poly',degree=3)
svmP1.fit(Xtrain,ytrain)
predP1=svmP1.predict(Xtest)
print("Classification report for classifier %s:\n%s\n"
      % (svmP1, metrics.classification_report(ytest,predP1)))

#Run kfold cross-validation to check cost parameters for polynomial kernel
precision=[]
cprecision=[]
Crange=np.logspace(-6,2,9)
for crange in Crange:
    kfold1=cross_validation.KFold(42000,n_folds=4)
    precision=[]
    for train,test in kfold1:
        Xtrain,Xtest,ytrain,ytest=X[train],X[test],y[train],y[test]
        svm1=svm.SVC(kernel='poly',degree=3,C=crange)
        svm1.fit(Xtrain,ytrain)
        ypred=svm1.predict(Xtest)
        precision.append(metrics.precision_score(ytest,ypred))
    cprecision.append(np.mean(precision))



#Use polynomial degree=3 for final model and submission
svm1=svm.SVC(kernel='poly',degree=3)
svm1.fit(X,y)
test=pd.read_csv('test.csv')
pred=svm1.predict(test)
pred = pd.DataFrame(pred)
pred['ImageId'] = pred.index + 1
pred = pred[['ImageId', 0]]
pred.columns = ['ImageId', 'Label']
pred.to_csv('pred.csv', index=False)



