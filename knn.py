import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

#Read training data and split into train and test data
data=pd.read_csv('train.csv')
data1=data.values
X=data1[:,1:]
y=data1[:,:1]
y=np.ravel(y)
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.25)

#Run PCA and KNN
pca=PCA(n_components=50).fit(Xtrain)
Xtrain_reduced=pca.transform(Xtrain)
Xtest_reduced=pca.transform(Xtest)
knn=KNeighborsClassifier(n_neighbors=5,weights='distance',p=3) 
knn.fit(Xtrain_reduced,ytrain)
pred=knn.predict(Xtest_reduced)

print("Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(ytest,pred)))
Classification report for classifier KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_neighbors=5, p=3, weights='distance'):
             precision    recall  f1-score   support

          0       0.98      0.99      0.99      1008
          1       0.97      0.99      0.98      1110
          2       0.98      0.97      0.98      1055
          3       0.97      0.96      0.96      1128
          4       0.98      0.97      0.98      1068
          5       0.97      0.97      0.97       950
          6       0.98      0.99      0.98      1025
          7       0.97      0.97      0.97      1096
          8       0.98      0.94      0.96      1017
          9       0.95      0.97      0.96      1043

avg / total       0.97      0.97      0.97     10500


#Run prediction on test data and make submissions
test=pd.read_csv('test.csv')
test_reduced=pca.transform(test)
pred2=knn.predict(test_reduced)
pred2 = pd.DataFrame(pred2)
pred2['ImageId'] = pred2.index + 1
pred2 = pred2[['ImageId', 0]]
pred2.columns = ['ImageId', 'Label']
pred2.to_csv('pred2.csv', index=False)
