import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from nltk.corpus import wordnet as wn
import lev

try:
    stopwords = stopwords.words('english')
except:
    stopwords = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                     'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                     'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                     'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                     'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                     'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                     'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                     'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])


def vectorizer(q1, q2, tmp_dict={}):
    t_1 = tokenoze(q1)
    t_2 = tokenoze(q2)
    tmp_dict["cosin"] = cosin(t_1, t_2)
    tmp_dict['scor_1-2'] = score(t_1, t_2)
    tmp_dict['scor_2-1'] = score(t_2, t_1)
    tmp_dict['exact_word'] = exact_word(t_1, t_2)
    tmp_dict['lev_dict'] = lev.levenshtein(t_1, t_2)
    total = len(t_1) + len(t_2) + 1
    tmp_dict['len_q1'] = len(t_1) / float(total)
    tmp_dict['len_q2'] = len(t_2) / float(total)
    tmp_dict['lch'] = list_sim(t_1, t_2, lch)
    tmp_dict['pats'] = list_sim(t_1, t_2, pats)
    tmp_dict['wup'] = list_sim(t_1, t_2, wup)
    tmp_dict['exact_word_distance'] = exact_word_distance(t_1, t_2)
    tmp_dict["match_kaggle"] = word_match_share(q1, q2)
    return tmp_dict


def str_lower(sentence):
    return str(sentence).lower()


def transform(df, weight=None):
    df['question1'] = df['question1'].apply(str_lower)
    df['question2'] = df['question2'].apply(str_lower)
    df['match_kaggle'] = df.apply(lambda x: word_match_share(x['question1'], x['question2']))
    df['t_1'] = df['question1'].apply(tokenoze)
    df['t_2'] = df['question2'].apply(tokenoze)
    df["cosin"] = df.apply(lambda x: cosin(x['t_1'], x['t_2']))
    df['scor_1-2'] = df.apply(lambda x: score(x['t_1'], x['t_2']))
    df['scor_2-1'] = df.apply(lambda x: score(x['t_2'], x['t_1']))
    df['exact_word'] = df.apply(lambda x: exact_word(x['t_1'], x['t_2']))
    df['lev_dict'] = df.apply(lambda x: lev.levenshtein(x['t_1'], x['t_2']))
    total = len(t_1) + len(t_2) + 1
    df['len_q1_word'] = df['t_1'].apply(len)  # (t_1)/float(total)
    df['len_q2_word'] = df['t_2'].apply(len)  # len(t_2)/float(total)
    df['lch'] = df.apply(lambda x: list_sim(x['t_1'], x['t_2'], lch))
    df['pats'] = df.apply(lambda x: list_sim(x['t_1'], x['t_2'], pats))
    df['wup'] = df.apply(lambda x: list_sim(x['t_1'], x['t_2'], wup))
    df['exact_word_distance'] = df.apply(lambda x: exact_word_distance(x['t_1'], x['t_2']))
    df['len_char_q1'] = df['question1'].apply(len)
    df['len_char_q2'] = df['question2'].apply(len)
    if weight:
        df['tf_idf'] = df.apply(lambda x: tfidf_word_match_share(x['question1'], x['question,weight'], weight))
    features = ['cosin', 'scor_1-2', 'scor_2-1', 'exact_word', 'lev_dict', "len_q1_word", "len_q2_word", "lch", "pats",
                "wup", 'exact_word_distance', 'len_char_q1', 'len_char_q2']

    return df[features]


def tfidf_word_match_share(question1, question2, weights):
    q1words = {}
    q2words = {}
    for word in str(question1).lower().split():
        if word not in stopwords:
            q1words[word] = 1
    for word in str(question2).lower().split():
        if word not in stopwords:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0

    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in
                                                                                    q2words.keys() if w in q1words]
    total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

    R = np.sum(shared_weights) / np.sum(total_weights)
    return R


def word_match_share(question1, question2):
    """ from kaggle https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb
    """
    q1words = {}
    q2words = {}
    for word in str(question1).lower().split():
        if word not in stopwords:
            q1words[word] = 1
    for word in str(question2).lower().split():
        if word not in stopwords:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(q1words) + len(q2words))
    return R


def lch(w1, w2):
    return wn.lch_similarity(wn.synsets(w1)[0], wn.synsets(w2)[0])


def pats(w1, w2):
    return wn.path_similarity(wn.synsets(w1)[0], wn.synsets(w2)[0])


def wup(w1, w2):
    return wn.wup_similarity(wn.synsets(w1)[0], wn.synsets(w2)[0])


def list_sim(W1, W2, func):
    res = 0.0
    counter = 1.0
    for w1 in W1:
        for w2 in W2:
            try:
                counter += 1
                res += func(w1, w2)
            except:
                res += 0
    return res / (counter)


def exact_word_distance(W1, W2):
    tmp = list(set(W1 + W2))
    size_ = max(len(W1), len(W2))
    tmp_w1 = [0] * size_
    tmp_w2 = [0] * size_
    tmp_ = [0] * size_
    for i, word in enumerate(W1):
        tmp_w1[i] = tmp.index(word)
    for i, word in enumerate(W2):
        tmp_w2[i] = (tmp.index(word) - tmp_w1[i])
        tmp_[i] = (tmp.index(word) + tmp_w1[i])
    return 1 - (np.linalg.norm(tmp_w2) / (np.linalg.norm(tmp_) + 1))


def cosin(t_1, t_2):
    tmp = {}
    tmp2 = {}
    vocab = []
    for word in t_1:
        if word not in tmp:
            tmp[word] = 1
        else:
            tmp[word] += 1
        if word not in vocab:
            vocab.append(word)

    for word in t_2:
        if word not in tmp2:
            tmp2[word] = 1
        else:
            tmp2[word] += 1
        if word not in vocab:
            vocab.append(word)
    for word in vocab:
        if word not in tmp:
            tmp[word] = 0
        if word not in tmp2:
            tmp2[word] = 0
    up = 0.0
    for key in tmp:
        up += tmp[key] * tmp2[key]
    res = up / (np.linalg.norm(tmp.values()) * np.linalg.norm(tmp2.values()) + 1)
    return res


def exact_word(t_1, t_2):
    res = 0.0
    res2 = 0.0
    for i, val in enumerate(t_1):
        if val in t_2:
            res += 1
    for i, val in enumerate(t_2):
        if val in t_1:
            res2 += 1
    return (res + res2 + 1) / (len(t_1) + len(t_2) + 1)


def tf(d):
    tmp = {}
    for word in d:
        if word not in tmp:
            tmp[word] = 1.0 / (len(d) + 1)
        else:
            tmp[word] += 1.0 / (len(d) + 1)
    return tmp


def df(tf_1, tf_2):
    tmp_df = {}
    for key in tf_1:
        if key not in tmp_df:
            tmp_df[key] = 1
    for key in tf_2:
        if key not in tmp_df:
            tmp_df[key] = 1
        else:
            tmp_df[key] += 1
    return tmp_df


def idf(t, df_, n_doc=2.0):
    return 1 + np.log(n_doc / (df_[t] + 1))


def queryNorm(tf_q, df_):
    return 1 / np.sqrt(sum([df_[key] ** 2 for key in tf_q]) + 1)


def norm(tf_d, t):
    if t not in tf_d:
        return 0
    return 1 / np.sqrt(tf_d[t])


def score(q, d):
    """custom score (from search engine)

    """
    tf_q = tf(q)
    tf_d = tf(d)
    df_ = df(tf_q, tf_d)
    res = 0.0
    for t in q:
        if t in d:
            res += tf_q[t] * (idf(t, df_) ** 2) * norm(tf_d, t)
    res = queryNorm(tf_q, df_) * res
    return res


def tokenoze(question):
    question = str(question).lower()
    t_1 = []
    try:
        for word in word_tokenize(question):
            if word not in stopwords:
                t_1.append(word)
    except:
        t_1.append("None")
    return t_1


def percentage(data_list):
    numberDoc = len(data_list)
    words = word_tokenize(' '.join([unicode(item, errors='replace') for item in data_list]))
    counts = Counter(words)
    tmp = {word: float(count) for word, count in counts.items()}
    total = sum(tmp.values())
    return {word: tmp[word] / total for word in tmp}


"""Some test
t_1 = ['d','r','i']
t_2 = ['d','r','t']
print Counter(t_1)
tf_1 = tf(t_1)
df_  = df(tf(t_1),tf(t_2))
print queryNorm(tf_1,df_)
print idf('d', df(tf(t_1),tf(t_2)))
print score(t_1,t_2)
print cosin(t_1,t_2)"""
