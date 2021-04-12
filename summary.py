from attention import AttentionLayer
from transformer import TextNormalizer

import numpy as np
import pandas as pd
import nltk
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras import backend as K
import warnings

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")


def text_cleaner(corpus):
    """
    Убирает знаки пунктуации и стоп-слова из корпуса
    """
    for text in corpus:
        new_string = text.lower()
        tokens = [w.strip('.,:;?') for w in new_string.split() if w not in stop_words and not TextNormalizer.is_punct(w)]
        yield (" ".join(tokens)).strip()


def start_end_tokens(text):
    """
    Добавление токенов начала и конца sol и eol
    """
    return np.array(pd.DataFrame({'p': text})['p'].apply(lambda x: 'sol ' + x + ' eol'))


if __name__ == 'main':
    data_train = pd.read_json("corpus/train.jsonl", lines=True)[:100]
    data_val = pd.read_json("corpus/val.jsonl", lines=True)[:25]
    data_test = pd.read_json("corpus/test.jsonl", lines=True)[:10]

    data_train.info()
    data_val.info()
    data_test.info()

    stop_words = set(nltk.corpus.stopwords.words('russian'))

    # Очистка корпуса
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
    text_word_count = []
    summary_word_count = []

    for i in data['cleaned_train_text']:
        text_word_count.append(len(i.split()))

    for i in data['cleaned_train_summary']:
        summary_word_count.append(len(i.split()))

    length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})

    length_df.hist(bins=30)
    plt.show()

    # На основании графиков и скорости обучения выбираем
    # Максимальную длину текста и реферата
    max_summary_len = 50
    max_text_len = 400

    # Выбор предложений которые меньше максимума
    cleaned_train_text = np.array(data['cleaned_train_text'])
    cleaned_train_summary = np.array(data['cleaned_train_summary'])

    cleaned_test_text = np.array(data['cleaned_test_text'])
    cleaned_test_summary = np.array(data['cleaned_test_summary'])

    cleaned_val_text = np.array(data['cleaned_val_text'])
    cleaned_val_summary = np.array(data['cleaned_val_summary'])

    # Здесь будут храниться предложения, отвечающию условиям максимума
    train_text = []
    train_summary = []
    for i in range(len(cleaned_train_text)):
        if len(cleaned_train_summary[i].split()) <= max_summary_len and len(cleaned_train_text[i].split()) <= max_text_len:
            train_text.append(cleaned_train_text[i])
            train_summary.append(cleaned_train_summary[i])

    test_text = []
    test_summary = []
    for i in range(len(cleaned_test_text)):
        if len(cleaned_test_summary[i].split()) <= max_summary_len and len(cleaned_test_text[i].split()) <= max_text_len:
            test_text.append(cleaned_test_text[i])
            test_summary.append(cleaned_test_summary[i])

    val_text = []
    val_summary = []
    for i in range(len(cleaned_val_text)):
        if len(cleaned_val_summary[i].split()) <= max_summary_len and len(cleaned_val_text[i].split()) <= max_text_len:
            val_text.append(cleaned_val_text[i])
            val_summary.append(cleaned_val_summary[i])

    # Добавление токенов начала и конца
    x_train = start_end_tokens(train_text)
    y_train = start_end_tokens(train_summary)
    x_val = start_end_tokens(val_text)
    y_val = start_end_tokens(val_summary)
    x_test = start_end_tokens(test_text)
    y_test = start_end_tokens(test_summary)

    # Токенизация
    x_tokenize = Tokenizer()
    x_tokenize.fit_on_texts(list(x_train))
    # Строки в числа
    x_train_seq = x_tokenize.texts_to_sequences(x_train)
    x_val_seq = x_tokenize.texts_to_sequences(x_val)
    # Добавление пустот
    x_tr = pad_sequences(x_train_seq, maxlen=max_text_len, padding='post')
    x_val = pad_sequences(x_val_seq, maxlen=max_text_len, padding='post')
    # Размер словаря (+1 для токена пустот)
    x_voc = x_tokenize.num_words + 1

    # Токенизатор для тренировочных данных
    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(list(y_train))
    # Строки в числа
    y_tr_seq = y_tokenizer.texts_to_sequences(y_train)
    y_val_seq = y_tokenizer.texts_to_sequences(y_val)
    # Добавление пустот
    y_tr = pad_sequences(y_tr_seq, maxlen=max_summary_len, padding='post')
    y_val = pad_sequences(y_val_seq, maxlen=max_summary_len, padding='post')
    # Размер словаря (+1 для токена пустот)
    y_voc = y_tokenizer.num_words +1

    # Удаление предложений, содержащих только SOL и EOL
    indexes = []
    for i in range(len(y_tr)):
        word_count = 0
        for j in y_tr[i]:
            if j != 0:
                word_count = word_count + 1
        if word_count == 2: # Только eol и sol
            indexes.append(i)
    y_tr = np.delete(y_tr, indexes, axis=0)
    x_tr = np.delete(x_tr, indexes, axis=0)

    # Тоже для тестовых
    indexes = []
    for i in range(len(y_val)):
        word_count = 0
        for j in y_val[i]:
            if j != 0:
                word_count = word_count + 1
        if(word_count == 2):
            indexes.append(i)
    y_val = np.delete(y_val, indexes, axis=0)
    x_val = np.delete(x_val, indexes, axis=0)

    # Построение модели
    K.clear_session()
    # Параметры измерений
    latent_dimensions = 300
    embedding_dimensions = 100
    # Кодировщик
    encoder_inputs_layer = Input(shape=(max_text_len,))
    # Встраиваемый слой
    encoder_embedding_layer = Embedding(x_voc, embedding_dimensions, trainable=True)(encoder_inputs_layer)
    # Кодировщик lstm 1
    encoder_lstm1_layer = LSTM(latent_dimensions, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_layer_output_1, _, _ = encoder_lstm1_layer(encoder_embedding_layer)
    # Кодировщик lstm 2
    encoder_lstm2_layer = LSTM(latent_dimensions, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4)
    encoder_layer_output_2, _, _ = encoder_lstm2_layer(encoder_layer_output_1)
    # Кодировщик lstm 3
    encoder_lstm3_layer = LSTM(latent_dimensions, return_state=True, return_sequences=True, dropout=0.4, recurrent_dropout=0.4)
    # state_h - Вывод скрытых состояний
    # state_c - Вывод состояния ячейки
    encoder_outputs, state_h, state_c = encoder_lstm3_layer(encoder_layer_output_2)
    # Настройка декодировщика
    decoder_inputs_layer = Input(shape=(None,))
    # Встраиваемый слой
    decoder_embedding_layer = Embedding(y_voc, embedding_dimensions, trainable=True)
    decoder_embedding = decoder_embedding_layer(decoder_inputs_layer)
    # Декодировщик lstm
    decoder_lstm_layer = LSTM(latent_dimensions, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2)
    decoder_outputs, _, _ = decoder_lstm_layer(decoder_embedding, initial_state=[state_h, state_c])
    # Слой внимания
    attention_layer = AttentionLayer(name='attention_layer')
    attention_out, attention_states = attention_layer([encoder_outputs, decoder_outputs])
    # Соединяем слой внимания и декодировщик LSTM
    decoder_concat_input_layer = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_out])
    # dense слой
    decoder_dense_layer = TimeDistributed(Dense(y_voc, activation='softmax'))
    decoder_outputs = decoder_dense_layer(decoder_concat_input_layer)
    # Определение модели
    model = Model([encoder_inputs_layer, decoder_inputs_layer], decoder_outputs)
    model.summary()
    # Компиляция модели
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

    # Освобождение памяти
    del data_train
    del data_val
    del data_test
    del summary_word_count
    del x_train_seq
    del y_tr_seq
    del y_val_seq
    del x_val_seq

    history = model.fit(
        [x_tr, y_tr[:, :-1]],
        y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:],
        epochs=4,
        callbacks=[es],
        batch_size=128,
        validation_data=(
            [x_val, y_val[:, :-1]],
            y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:])
    )
    # График обучениия
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    # Отсюда начинается тест модели
    # Словари для преобразования индексов в слова
    sum_index_word = y_tokenizer.index_word
    text_index_word = x_tokenize.index_word
    sum_word_index = y_tokenizer.word_index
    # Кодирование входной последовательности, настройка вывода
    encoder_model = Model(inputs=encoder_inputs_layer, outputs=[encoder_outputs, state_h, state_c])
    # Настройка декодировщика
    # Будут сохранять состояния с предыдущих временных шагов
    decoder_input_h = Input(shape=(latent_dimensions,))
    decoder_input_c = Input(shape=(latent_dimensions,))
    decoder_hidden_state_input = Input(shape=(max_text_len, latent_dimensions))
    # Получение вложений последовательности с декодеровщика
    decoder_embedding_2 = decoder_embedding_layer(decoder_inputs_layer)
    # Для предсказания следующего слова в последовательности
    # устанавливаю начальные состояния равными состояниям с предыдущего временного шага
    decoder_outputs_2, state_h2, state_c2 = decoder_lstm_layer(decoder_embedding_2, initial_state=[decoder_input_h, decoder_input_c])
    # Вывод со слоя внимания
    attention_out_inf, attention_states_inf = attention_layer([decoder_hidden_state_input, decoder_outputs_2])
    decoder_concat = Concatenate(axis=-1, name='concat')([decoder_outputs_2, attention_out_inf])
    # Слой softmax
    decoder_outputs_2 = decoder_dense_layer(decoder_concat)
    # Финальная модель декодеровщика
    decoder_model = Model(
        [decoder_inputs_layer] + [decoder_hidden_state_input, decoder_input_h, decoder_input_c],
        [decoder_outputs_2] + [state_h2, state_c2])

    for i in range(0,100):
        print("Review:",seq2text(x_tr[i]))
        print("Original summary:",seq2summary(y_tr[i]))
        print("Predicted summary:",decode_sequence(x_tr[i].reshape(1,max_text_len)))
        print("\n")


def decode_sequence(input_seq):
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, 1))

    target_seq[0, 0] = sum_word_index['sol']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:

        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = sum_index_word[sampled_token_index]

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
        if((i != 0 and i != sum_word_index['sol']) and i!=sum_word_index['eol']):
            newString= newString + sum_index_word[i] + ' '
    return newString


def seq2text(input_seq):
    newString=''
    for i in input_seq:
        if(i!=0):
            newString= newString + text_index_word[i] + ' '
    return newString


