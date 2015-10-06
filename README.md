# kagge-digit-recognition

# My code:

	randomforest.py for RandomForestClassifier
	knn.py for knn and pca
	svm.py for svm
Best score is 0.97543.
It is my second kaggle competitions. It is a lot of fun to try different algorithms, tuning parameters, and using cross-validation.

# First attempt with RandomForestClassifier

Without tuning any parameters, and using train_test_split to cross-validate results, RandomForestClassifier achieved a precision score of 0.93 for testing data.

# Second attempt with KNN

It is important to use PCA to increase running speed. Again use train_test_split to cross-validate results. Choose pca(n_components=50)and KNeighborsClassifier(weights = 'distance', n_neighbors=5, p=3). The score is 0.97173.

# Third attempt with SVM

Use LinearSVC first, with various C values, and the precision score is around 0.90
Use RBF kernels next. For some reason, the running time is extremely long and the result is poor.
Use polynomial kernels and get good scores with degree=2 or 3. Use kfold cross validation and choose svm.SVC(kernel='poly',degree=3). The score is 0.97543, slight improvement over KNN.
