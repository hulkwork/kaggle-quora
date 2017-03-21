import pandas as pd
import sys

reload(sys)
sys.setdefaultencoding("utf_8")
import lev
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import utils
import datetime
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

# define new train data
data_train = []

t_init = datetime.datetime.now()

batch = 10000
size_train = len(train)
t_mean = t_init - t_init
for item in train.iterrows():
    tmp_dict = {}
    if True:
        tmp = item[1].to_dict()

        question1 = str(tmp['question1']).lower()
        question2 = str(tmp['question2']).lower()
        target = tmp['is_duplicate']
        tmp_dict = utils.vectorizer(question1, question2, tmp_dict,weights)
        tmp_dict['target'] = target
        data_train.append(tmp_dict)

    if len(data_train) % batch == 0:
        print "iteration data %d" % (len(data_train),)
        t_mean += (datetime.datetime.now() - t_init)
        tmp = {'time': str(datetime.datetime.now() - t_init), 'batch': batch, 'left': size_train - len(data_train)}
        iteration = len(data_train) // batch
        t_mean_tmp = t_mean / iteration
        tmp['tpred'] = (tmp['left'] // batch) * t_mean_tmp
        tmp['tpred'] = str(tmp['tpred'])

        print "second per {batch} record(s) {time} prediction for {left} left records : {tpred}".format(**tmp)
        t_init = datetime.datetime.now()

d = pd.DataFrame(data_train)
# save for future prediction
d.to_csv('train_distance.csv', index=False)

import update
