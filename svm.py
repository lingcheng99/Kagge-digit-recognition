import numpy as np
import pandas as pd
from sklearn import metrics,cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn import svm

#Read training data and split into train and test data
data=pd.read_csv('train.csv')
data1=data.values
X=data1[:,1:]
y=data1[:,:1]
y=np.ravel(y)
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.5)

#Run linear kernel first
svmL1=svm.SVC(kernel='linear',C=0.01)
svmL1.fit(Xtrain,ytrain) 
predL1=svmL1.predict(Xtest) 
print("Classification report for classifier %s:\n%s\n"
      % (svmL1, metrics.classification_report(ytest,predL1)))
Classification report for classifier SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='linear', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False):
             precision    recall  f1-score   support

          0       0.95      0.97      0.96      2101
          1       0.95      0.98      0.96      2386
          2       0.89      0.91      0.90      2092
          3       0.85      0.89      0.87      2126
          4       0.89      0.93      0.91      2007
          5       0.87      0.87      0.87      1881
          6       0.96      0.94      0.95      2116
          7       0.93      0.91      0.92      2202
          8       0.92      0.83      0.87      2018
          9       0.89      0.86      0.87      2071

avg / total       0.91      0.91      0.91     21000

#Use GridSearchCV to tune parameters.C=1e-06 gave the best result.
parameters3={'kernel':['linear'],'C': [1e-06, 0.0001, 0.01, 1, 100]}
clf3=GridSearchCV(svm.SVC(),parameters3,cv=5,scoring='precision')
clf3.fit(Xtrain,ytrain)
clf3.grid_scores_

Out[17]:
[mean: 0.93872, std: 0.00203, params: {'kernel': 'linear', 'C': 1e-06},
 mean: 0.91670, std: 0.00370, params: {'kernel': 'linear', 'C': 0.0001},
 mean: 0.91276, std: 0.00306, params: {'kernel': 'linear', 'C': 0.01},
 mean: 0.91276, std: 0.00306, params: {'kernel': 'linear', 'C': 1},
 mean: 0.91276, std: 0.00306, params: {'kernel': 'linear', 'C': 100}]


#Run gaussian kernel but the result is poor and running time is long
svmR1=svm.SVC(kernel='rbf',gamma=0.001, C=10000)
svmR1.fit(Xtrain,ytrain)
predR1=svmR1.predict(Xtest)
print("Classification report for classifier %s:\n%s\n"
      % (svmR1, metrics.classification_report(ytest,predR1)))
Classification report for classifier SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0, degree=3,
  gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
  random_state=None, shrinking=True, tol=0.001, verbose=False):
             precision    recall  f1-score   support

          0       0.00      0.00      0.00      2101
          1       0.11      1.00      0.20      2386
          2       0.00      0.00      0.00      2092
          3       0.00      0.00      0.00      2126
          4       0.00      0.00      0.00      2007
          5       0.00      0.00      0.00      1881
          6       0.00      0.00      0.00      2116
          7       0.00      0.00      0.00      2202
          8       0.00      0.00      0.00      2018
          9       0.00      0.00      0.00      2071

avg / total       0.01      0.11      0.02     21000



#Run polynomial kernel
svmP1=svm.SVC(kernel='poly',degree=3)
svmP1.fit(Xtrain,ytrain)
predP1=svmP1.predict(Xtest)
print("Classification report for classifier %s:\n%s\n"
      % (svmP1, metrics.classification_report(ytest,predP1)))
Classification report for classifier SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3, gamma=0.0,
  kernel='poly', max_iter=-1, probability=False, random_state=None,
  shrinking=True, tol=0.001, verbose=False):
             precision    recall  f1-score   support

          0       0.97      0.98      0.98      2101
          1       0.97      0.99      0.98      2386
          2       0.97      0.96      0.96      2092
          3       0.96      0.96      0.96      2126
          4       0.97      0.97      0.97      2007
          5       0.96      0.96      0.96      1881
          6       0.98      0.97      0.98      2116
          7       0.98      0.97      0.97      2202
          8       0.97      0.96      0.96      2018
          9       0.97      0.95      0.96      2071

avg / total       0.97      0.97      0.97     21000


#Use GridSearchCV to tune parameters.Degree=2 gave the best result.
parameters1={'kernel':['poly'],'degree':[2,3,4]}
clf1=GridSearchCV(svm.SVC(),parameters1,cv=5,scoring='precision')
clf1.fit(Xtrain,ytrain)
clf1.grid_scores_

Out[13]:
[mean: 0.96890, std: 0.00264, params: {'kernel': 'poly', 'degree': 2},
 mean: 0.96567, std: 0.00233, params: {'kernel': 'poly', 'degree': 3},
 mean: 0.95507, std: 0.00210, params: {'kernel': 'poly', 'degree': 4}]

#Use polynomial degree=2 for final model and submission
svm1=svm.SVC(kernel='poly',degree=2)
svm1.fit(X,y)
test=pd.read_csv('test.csv')
pred=svm1.predict(test)
pred = pd.DataFrame(pred)
pred['ImageId'] = pred.index + 1
pred = pred[['ImageId', 0]]
pred.columns = ['ImageId', 'Label']
pred.to_csv('pred.csv', index=False)



