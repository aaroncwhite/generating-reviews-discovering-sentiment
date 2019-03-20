import os
import html
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('ggplot')

class SentimentResult(object):
    """A sentiment result object with an overall sentiment score
    and individual character level results.
    """

    def __init__(self, text, sentiment, char_sentiment, n_limit=64):
        """Initializes a results object to store text, sentiment score
        and character level scores together in one place.
        
        Arguments:
            text {str} -- A string that was encoded by the model
            sentiment {float} -- A sentiment score
            char_sentiment {list of float} -- list of floats describing 
                sentiment of the character sequence

        """ 
        self.text = text
        self.sentiment = sentiment
        self.char_sentiment = char_sentiment
        self.n_limit = n_limit
    
    def __repr__(self):
        return repr(self.sentiment)

    def to_dict(self):
        return self.__dict__

    def plot(self, filename=None):
        values = self.char_sentiment.flatten()
        preprocessed_text = self.text
        n_limit = self.n_limit
        num_chars = len(preprocessed_text)

        i = 0
        while i < num_chars:
 
            if i + n_limit > num_chars:
                end_index = num_chars
                # We've reached a shorter than n_limit number of 
                # characters to display, so pad the output results with
                # empty strings and 0s so we can make the plot look even
                n_missing = n_limit - (end_index % n_limit)
                add_chars = ' ' * n_missing
                add_points = np.zeros((n_missing,))

            else:
                # Set defaults here so the plotting process does not break
                end_index = i+n_limit
                add_chars = ''
                add_points = []
            
            values_limited = np.concatenate([values[i:end_index], add_points])
            data = values_limited.reshape((1, min([n_limit, len(values_limited)])))
            labels = np.array([x for x in preprocessed_text[i:end_index] + add_chars]).reshape((1, min([n_limit, len(values_limited)])))
            fig, ax = plt.subplots(figsize=(20,0.5))
            ax = sns.heatmap(data, annot = labels, fmt = '', annot_kws={"size":15}, vmin=-1, vmax=1, cmap='RdYlGn')
            
            i+=n_limit
            

        if filename:
            fig.savefig(filename)
        fig = plt.figure()
        plt.close()
        return fig




def train_with_reg_cv(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
        C=2**np.arange(-8, 1).astype(np.float), seed=42):
    scores = []
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i)
        model.fit(trX, trY)
        score = model.score(vaX, vaY)
        scores.append(score)
    c = C[np.argmax(scores)]
    model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C))
    model.fit(trX, trY)
    nnotzero = np.sum(model.coef_ != 0)
    if teX is not None and teY is not None:
        score = model.score(teX, teY)*100.
    else:
        score = model.score(vaX, vaY)*100.
    return score, c, nnotzero


def load_sst(path):
    data = pd.read_csv(path)
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y


def sst_binary(data_dir='data/'):
    """
    Most standard models make use of a preprocessed/tokenized/lowercased version
    of Stanford Sentiment Treebank. Our model extracts features from a version
    of the dataset using the raw text instead which we've included in the data
    folder.
    """
    trX, trY = load_sst(os.path.join(data_dir, 'train_binary_sent.csv'))
    vaX, vaY = load_sst(os.path.join(data_dir, 'dev_binary_sent.csv'))
    teX, teY = load_sst(os.path.join(data_dir, 'test_binary_sent.csv'))
    return trX, vaX, teX, trY, vaY, teY


def find_trainable_variables(key):
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))


def preprocess(text, front_pad='\n ', end_pad=' '):
    text = html.unescape(text)
    text = text.replace('\n', ' ').strip()
    text = front_pad+text+end_pad
    text = text.encode()
    return text


def iter_data(*data, **kwargs):
    size = kwargs.get('size', 128)
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


class HParams(object):

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
