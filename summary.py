from attention import AttentionLayer
from transformer import TextNormalizer

import numpy as np
import pandas as pd
import nltk
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

data_train = pd.read_json("corpus/train.jsonl", lines=True)[:100]
data_val = pd.read_json("corpus/val.jsonl", lines=True)[:25]
data_test = pd.read_json("corpus/test.jsonl", lines=True)[:10]

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
max_text_len = 400

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

x_train = np.array(pd.DataFrame({'p': train_text})['p'].apply(lambda x: 'sol ' + x + ' eol'))
y_train = np.array(pd.DataFrame({'p': train_summary})['p'].apply(lambda x: 'sol ' + x + ' eol'))


x_val = np.array(pd.DataFrame({'p': val_text})['p'].apply(lambda x: 'sol ' + x + ' eol'))
y_val = np.array(pd.DataFrame({'p': val_summary})['p'].apply(lambda x: 'sol ' + x + ' eol'))

x_test = np.array(pd.DataFrame({'p': test_text})['p'].apply(lambda x: 'sol ' + x + ' eol'))
y_test = np.array(pd.DataFrame({'p': test_summary})['p'].apply(lambda x: 'sol ' + x + ' eol'))

# Токенизация
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_train))

# TODO: Возможно, на 4 поменять, если медленно обучаться будет
thresh = 1

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

thresh = 1

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

print((y_tokenizer.word_counts['sol'], len(y_tr)))


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
from keras import backend as K
K.clear_session()

latent_dim = 300
embedding_dim = 100

# Кодировщик
encoder_inputs = Input(shape=(max_text_len,))

# embedding layer
enc_emb = Embedding(x_voc, embedding_dim, trainable=True)(encoder_inputs)

# encoder lstm 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# encoder lstm 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# encoder lstm 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# Настройка декодировщика
decoder_inputs = Input(shape=(None,))

# embedding layer
dec_emb_layer = Embedding(y_voc, embedding_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
decoder_outputs,decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# слой Attention
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Соединяем слой внимания и декодировщик LSTM
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# dense слой
decoder_dense = TimeDistributed(Dense(y_voc, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Определение модели
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)

del data_train
del data_val
del data_test
del summary_word_count

del x_tr_seq

del y_tr_seq
del y_val_seq
del x_val_seq

history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:] ,epochs=4,callbacks=[es],batch_size=128, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))


from matplotlib import pyplot
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_text_len,latent_dim))

dec_emb2= dec_emb_layer(decoder_inputs)

decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])


attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])


decoder_outputs2 = decoder_dense(decoder_inf_concat)


decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = target_word_index['sol']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index]

        if sampled_token != 'eol':
            decoded_sentence += ' ' + sampled_token

        if sampled_token == 'eol' or len(decoded_sentence.split()) >= (max_summary_len - 1):
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    newString=''
    for i in input_seq:
        if((i!=0 and i!=target_word_index['sol']) and i!=target_word_index['eol']):
            newString=newString+reverse_target_word_index[i]+' '
    return newString


def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString=newString+reverse_source_word_index[i]+' '
    return newString


for i in range(0,100):
    print("Review:",seq2text(x_tr[i]))
    print("Original summary:",seq2summary(y_tr[i]))
    print("Predicted summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))
    print("\n")