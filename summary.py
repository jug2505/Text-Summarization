from attention import AttentionLayer
from transformer import TextNormalizer

import numpy as np
import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

data_train = pd.read_json("corpus/train.jsonl", lines=True)
data_val = pd.read_json("corpus/val.jsonl", lines=True)
data_test = pd.read_json("corpus/test.jsonl", lines=True)

data_train.info()
data_val.info()
data_test.info()

stop_words = set(nltk.corpus.stopwords.words('russian'))


def text_cleaner(corpus):
    for text in corpus:
        new_string = text.lower()
        tokens = [w.strip('.,') for w in new_string.split() if w not in stop_words and not TextNormalizer.is_punct(w)]
        yield (" ".join(tokens)).strip()


data = {}

data_buff = []
for cleaned_train_text in text_cleaner(data_train['text']):
    data_buff.append(cleaned_train_text)
data['cleaned_train_text'] = data_buff

data_buff = []
for cleaned_train_summary in text_cleaner(data_train['summary']):
    data_buff.append(cleaned_train_summary)
data['cleaned_train_summary'] = data_buff

data_buff = []
for cleaned_test_text in text_cleaner(data_test['text']):
    data_buff.append(cleaned_test_text)
data['cleaned_test_text'] = data_buff

data_buff = []
for cleaned_test_summary in text_cleaner(data_test['summary']):
    data_buff.append(cleaned_test_summary)
data['cleaned_test_summary'] = data_buff

data_buff = []
for cleaned_val_text in text_cleaner(data_val['text']):
    data_buff.append(cleaned_val_text)
data['cleaned_val_text'] = data_buff

data_buff = []
for cleaned_val_summary in text_cleaner(data_val['summary']):
    data_buff.append(cleaned_val_summary)
data['cleaned_val_summary'] = data_buff


# Выбор максимальной длины
import matplotlib.pyplot as plt

text_word_count = []
summary_word_count = []

for i in data['cleaned_train_text']:
    text_word_count.append(len(i.split()))

for i in data['cleaned_train_summary']:
    summary_word_count.append(len(i.split()))

length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})

length_df.hist(bins=30)
plt.show()

max_summary_len = 50
max_text_len = 800

# Выбор предложений которые меньше максимума
cleaned_train_text = np.array(data['cleaned_train_text'])
cleaned_train_summary = np.array(data['cleaned_train_summary'])

cleaned_test_text = np.array(data['cleaned_test_text'])
cleaned_test_summary = np.array(data['cleaned_test_summary'])

cleaned_val_text = np.array(data['cleaned_val_text'])
cleaned_val_summary = np.array(data['cleaned_val_summary'])

test_text = []
test_summary = []

val_text = []
val_summary = []

train_text = []
train_summary = []

for i in range(len(cleaned_train_text)):
    if len(cleaned_train_summary[i].split()) <= max_summary_len and \
            len(cleaned_train_text[i].split()) <= max_text_len:
        train_text.append(cleaned_train_text[i])
        train_summary.append(cleaned_train_summary[i])

for i in range(len(cleaned_test_text)):
    if len(cleaned_test_summary[i].split()) <= max_summary_len and \
            len(cleaned_test_text[i].split()) <= max_text_len:
        test_text.append(cleaned_test_text[i])
        test_summary.append(cleaned_test_summary[i])

for i in range(len(cleaned_val_text)):
    if len(cleaned_val_summary[i].split()) <= max_summary_len and \
            len(cleaned_val_text[i].split()) <= max_text_len:
        val_text.append(cleaned_val_text[i])
        val_summary.append(cleaned_val_summary[i])

x_train = np.array(pd.DataFrame({'p': train_text})['p'].apply(lambda x: 'SOL ' + x + ' EOL'))
y_train = np.array(pd.DataFrame({'p': train_summary})['p'].apply(lambda x: 'SOL ' + x + ' EOL'))

x_val = np.array(pd.DataFrame({'p': val_text})['p'].apply(lambda x: 'SOL ' + x + ' EOL'))
y_val = np.array(pd.DataFrame({'p': val_summary})['p'].apply(lambda x: 'SOL ' + x + ' EOL'))

x_test = np.array(pd.DataFrame({'p': test_text})['p'].apply(lambda x: 'SOL ' + x + ' EOL'))
y_test = np.array(pd.DataFrame({'p': test_summary})['p'].apply(lambda x: 'SOL ' + x + ' EOL'))
