from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
import pandas as pd
import utils
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import lev
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import utils
train = pd.read_csv("train_distance.csv")

import xgboost as xgb

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.022
params['max_depth'] = 4




y = list(train['target'])
X = train.drop(['target'],axis=1)
X = X.fillna(-1)
n=0.8
size = len(X)
X_train = X[:int(n*size)] 
X_test = X[int(n*size)+1:]
y_train = y[:int(n*size)]
y_test = y[int(n*size)+1:]


d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_test, label=y_test)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 300, watchlist, early_stopping_rounds=50, verbose_eval=10)


import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import lev
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import utils

porter = nltk.PorterStemmer()
lancaster = nltk.LancasterStemmer()
wnl = nltk.WordNetLemmatizer()
test = pd.read_csv('test.csv')

exact_word = utils.exact_word


head = 'cosin,exact_word,scor_1-2,scor_2-1'.split(',')
test_id = []
prediction = []
counter = 0
ids = test['test_id']

test['t_1'] = test['question1'].apply(utils.tokenoze)
test['t_2'] = test['question2'].apply(utils.tokenoze)
test['cosin'] = test.apply(lambda x : utils.cosin(x['t_1'],x['t_2']))
test['scor_1-2'] = test.apply(lambda x : utils.score(x['t_1'],x['t_2']))
test['scor_2-1'] = test.apply(lambda x : utils.score(x['t_1'],x['t_2']))
test['exact_word'] = test.apply(lambda x : exact_word(x['t_1'],x['t_2']))
test['lev_dict'] = test.apply(lambda x : lev.levenshtein(x['t_1'],x['t_2']))
x_test = test[['cosin','scor_1-2','scor_2-1','exact_word','lev_dict']]
x_test.tocsv('test_distnace.csv',index=False)
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)
#pred = rfr.predict(x_test)[0]
d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)
print p_test
sub = pd.DataFrame()
sub['test_id'] = ids
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb_new.csv', index=False)
