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

# Токенизация
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_train))

# TODO: Возможно, на 4 поменять, если медленно обучаться будет
thresh = 4

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in x_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

print("% редких слов в словаре: ", (cnt / tot_cnt) * 100)
print("Общее покрытие редкими словами: ", (freq / tot_freq) * 100)

x_tokenizer = Tokenizer(num_words=tot_cnt-cnt)
x_tokenizer.fit_on_texts(list(x_train))

x_tr_seq = x_tokenizer.texts_to_sequences(x_train)
x_val_seq = x_tokenizer.texts_to_sequences(x_val)

x_tr = pad_sequences(x_tr_seq,  maxlen=max_text_len, padding='post')
x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')

x_voc = x_tokenizer.num_words + 1

print(x_voc)


# Токенизатор для тренировочных данных
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_train))

thresh = 6

cnt = 0
tot_cnt = 0
freq = 0
tot_freq = 0

for key, value in y_tokenizer.word_counts.items():
    tot_cnt = tot_cnt + 1
    tot_freq = tot_freq + value
    if (value < thresh):
        cnt = cnt + 1
        freq = freq + value

print("% редких слов в словаре:", (cnt / tot_cnt) * 100)
print("Общее покрытие редкими словами:", (freq / tot_freq) * 100)

y_tokenizer = Tokenizer(num_words=tot_cnt-cnt)
y_tokenizer.fit_on_texts(list(y_train))

y_tr_seq = y_tokenizer.texts_to_sequences(y_train)
y_val_seq = y_tokenizer.texts_to_sequences(y_val)

y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')

y_voc = y_tokenizer.num_words +1

print((y_tokenizer.word_counts['SOL'], len(y_tr)))


# Удаление предложений, содержащих только SOL и EOL

ind = []
for i in range(len(y_tr)):
    cnt = 0
    for j in y_tr[i]:
        if j != 0:
            cnt = cnt+1
    if(cnt == 2):
        ind.append(i)

y_tr = np.delete(y_tr, ind, axis=0)
x_tr = np.delete(x_tr, ind, axis=0)

ind = []
for i in range(len(y_val)):
    cnt = 0
    for j in y_val[i]:
        if j != 0:
            cnt = cnt+1
    if(cnt == 2):
        ind.append(i)

y_val = np.delete(y_val, ind, axis=0)
x_val = np.delete(x_val, ind, axis=0)

# Построение модели
