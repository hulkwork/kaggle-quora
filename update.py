from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from nltk import word_tokenize
import pandas as pd
import sys
import utils
import datetime 
reload(sys)
sys.setdefaultencoding("utf-8")

from collections import Counter
# Load train
train = pd.read_csv('train.csv', encoding='utf-8')
train_qs = pd.Series(train['question1'].tolist() + train['question2'].tolist()).astype(str)

words = (" ".join(train_qs)).lower().split()
count_words = len(words)
print "all words %d" % (count_words)
counts = Counter(words)

weights = {}
for word, count in counts.items():
    weights[word] = float(count) / count_words

train = pd.read_csv("train_distance.csv")

# Use random forest regressor

# version of sklearn use 0.14
y = list(train['target'])
X = train.drop(['target'],axis=1)
X = X.fillna(-1)

"""rfr = RandomForestRegressor(n_estimators=500,verbose=1, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, n_jobs=-1)

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

print "current prediction rate %s" % (str(float(mse)/(i+1)),)"""


import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.19
params['max_depth'] = 16
size = len(train)
n = 0.2
x_train = X[:int(n*size)] 
x_valid = X[int(n*size)+1:]
y_train = y[:int(n*size)]
y_valid = y[int(n*size)+1:]

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 1000, watchlist, early_stopping_rounds=150, verbose_eval=10)

test = pd.read_csv('test.csv')

head = []
for key in train.columns:
    if key !='target':
        head.append(key)
test_id = []
prediction = []
counter = 1
#sub = open('sub.csv','wb')
#sub.write('test_id,is_duplicate\n')
# prediction by item to avoid out of memory
ids = []
data_test = []
frames = []
t_init = datetime.datetime.now()
batch = 1000
size_test = len(test)
t_mean = t_init-t_init
for item in test.iterrows():
    counter += 1
    if True:
        tmp_dict = {}
        tmp = item[1].to_dict()
        question1 = str(tmp['question1']).lower()
        question1 = unicode(question1, errors='replace')
        question2 = str(tmp['question2']).lower()
        question2 = unicode(question2, errors='replace')
        tmp_dict = utils.vectorizer(question1, question2, tmp_dict,weights)
        data_test.append(tmp_dict)
        ids.append(tmp['test_id'])
        t_1 =   []      
        t_2 = []
        x_test = [tmp_dict[h] for h in head]
        #pred = rfr.predict(x_test)[0]
        #sub.write("{test_id},{is_duplicate}\n".format(**{"test_id":tmp['test_id'],"is_duplicate":pred}))

    if counter%batch == 0:
        x_test = pd.DataFrame(data_test)
        data_test = []

        d_test = xgb.DMatrix(x_test)
        p_test = bst.predict(d_test)
        sub = pd.DataFrame()
        sub['test_id'] = ids
        ids = []
        sub['is_duplicate'] = p_test
        #sub.to_csv('simple_xgb_%d.csv'%(counter), index=False)
        frames.append(sub)
        print "iteration data %d size of data %d" %(counter,len(p_test),)   
        t_mean += (datetime.datetime.now()-t_init )
        tmp = {'time':str(datetime.datetime.now()-t_init),'batch':batch,'left':size_test - counter}
        iteration = counter//batch +1
        t_mean_tmp = t_mean / iteration
        tmp['tpred'] = (tmp['left']//batch) * t_mean_tmp
        tmp['tpred'] = str(tmp['tpred'])

        print "second per {batch} record(s) {time} prediction for {left} left records : {tpred}" .format( **tmp)
        t_init = datetime.datetime.now()
x_test = pd.DataFrame(data_test)
data_test = []

d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)
sub = pd.DataFrame()
sub['test_id'] = ids
ids = []
sub['is_duplicate'] = p_test
        #sub.to_csv('simple_xgb_%d.csv'%(counter), index=False)
frames.append(sub)
print "iteration data %d size of data %d" %(counter,len(p_test),)
t_mean += (datetime.datetime.now()-t_init )
tmp = {'time':str(datetime.datetime.now()-t_init),'batch':batch,'left':size_test - counter}
iteration = counter//batch +1
t_mean_tmp = t_mean / iteration
tmp['tpred'] = (tmp['left']//batch) * t_mean_tmp
tmp['tpred'] = str(tmp['tpred'])

print "second per {batch} record(s) {time} prediction for {left} left records : {tpred}" .format( **tmp)
t_init = datetime.datetime.now()
result = pd.concat(frames)
result.to_csv('xgb_features.csv',index=False)
