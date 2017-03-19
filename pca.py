from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE

from sklearn.cross_validation import KFold

def MSE(rfr,X_test,y_test):
	mse = 0.0
	y_test = list(y_test)
	for i,val in enumerate(rfr.predict(X_test)):
		mse += (val-y_test[i])**2
	return np.sqrt(mse/len(y_test))

data=pd.read_csv('train_distance.csv')
y = data['target']
data = data.drop(['target'],axis=1)

X = data
pca = PCA(n_components=2)
X_r = pca.fit(X,y).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))
plt.figure()
colors = ['navy', 'darkorange']
lw = 2
target_names = ['is_duplicate','no']

"""for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)"""
kf = KFold(len(X_r), n_folds=5)
rfr = RandomForestRegressor(n_estimators=30,verbose=0, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, n_jobs=1)
for train_index, test_index in kf:
	print("TRAIN:", train_index, "TEST:", test_index)
	#X_r = pca.fit(X[train_index-1],y[train_index]).transform(X)
	X_train, X_test = X_r[train_index], X_r[test_index]
	y_train, y_test = y[train_index], y[test_index]
	rfr.fit(X_train,y_train)
	
	plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title('PCA of dataset score MSE : %s' % (  str(MSE( rfr,X_test,y_test )),) )
	plt.show()
	plt.clf()

