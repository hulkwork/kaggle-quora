from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from nltk import word_tokenize
import pandas as pd
import sys
import utils

reload(sys)
sys.setdefaultencoding("utf-8")

train = pd.read_csv("train_distance.csv")

# Use random forest regressor
rfr = RandomForestRegressor(n_estimators=30,verbose=0, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, n_jobs=1)

# version of sklearn use 0.14
y = list(train['target'])
X = train.drop(['target'],axis=1)
X = X.fillna(-1)
n=0.7
size = len(X)
X_train = X[:int(n*size)] 
X_test = X[int(n*size)+1:]
y_train = y[:int(n*size)]
y_test = y[int(n*size)+1:]

rfr.fit(X_train,y_train)

mse = 0.0
# Not the target metric (it's log-loss)
for i,val in enumerate(rfr.predict(X_test)):
	mse += (val-y_test[i])**2

print "current prediction rate %s" % (str(float(mse)/(i+1)),)

test = pd.read_csv('test.csv')

head = []
for key in train.columns:
	if key !='target':
		head.append(key)
test_id = []
prediction = []
counter = 0
sub = open('sub.csv','wb')
sub.write('test_id,is_duplicate\n')
# prediction by item to avoid out of memory
for item in test.iterrows():
	counter += 1
	if True:
		tmp_dict = {}
		tmp = item[1].to_dict()
		question1 = str(tmp['question1']).lower()
		question1 = unicode(question1, errors='replace')
		question2 = str(tmp['question2']).lower()
		question2 = unicode(question2, errors='replace')
		tmp_dict = utils.vectorizer(question1,question2,tmp_dict)
		t_1	=	[]		
		t_2 = []
		x_test = [tmp_dict[h] for h in head]
		pred = rfr.predict(x_test)[0]
		sub.write("{test_id},{is_duplicate}\n".format(**{"test_id":tmp['test_id'],"is_duplicate":pred}))

	if counter % 2000 == 0:
		print "iteration data %d" %(counter,)	
