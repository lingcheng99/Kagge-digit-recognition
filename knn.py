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

#Run prediction on test data and make submissions
test=pd.read_csv('test.csv')
test_reduced=pca.transform(test)
pred2=knn.predict(test_reduced)
pred2 = pd.DataFrame(pred2)
pred2['ImageId'] = pred2.index + 1
pred2 = pred2[['ImageId', 0]]
pred2.columns = ['ImageId', 'Label']
pred2.to_csv('pred2.csv', index=False)
