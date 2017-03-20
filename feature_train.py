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

# Load train
train = pd.read_csv('train.csv',encoding='utf-8')
# define new train data
data_train = []

t_init = datetime.datetime.now()
train = utils.transform(train)
train.to_csv('train_features.csv',index=False)
tmp = {"batch":len(train),'time':datetime.datetime.now()-t_init}
print "second for {batch} record(s) {time} " .format( **tmp)

