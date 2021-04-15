from keras.layers import Embedding, Dense, RepeatVector, TimeDistributed
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os


class SummarizerRNN(object):
    model_name = 'summarizer-rnn'

    def __init__(self, settings):
        self.number_input_words = settings['number_input_words']
        self.max_length_input = settings['max_length_input']
        self.number_output_words = settings['number_output_words']
        self.max_length_output = settings['max_length_output']
        self.input_word_to_index = settings['input_word_to_index']
        self.input_index_to_word = settings['input_index_to_word']
        self.output_word_to_index = settings['output_word_to_index']
        self.output_index_to_word = settings['output_index_to_word']
        self.settings = settings

        print('max_length_input', self.max_length_input)
        print('max_length_output', self.max_length_output)
        print('number_input_words', self.number_input_words)
        print('number_output_words', self.number_output_words)

        # Вход кодировщика
        model = Sequential()
        model.add(Embedding(output_dim=128, input_dim=self.number_input_words, input_length=self.max_length_input))

        # Модель кодировщика
        model.add(LSTM(128))
        model.add(RepeatVector(self.max_length_output))
        # Модель декодера
        model.add(LSTM(128, return_sequences=True))
        model.add(TimeDistributed(Dense(self.number_output_words, activation='softmax')))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model = model

    def transform_input_text(self, texts):
        temp = []
        for line in texts:
            x = []
            for word in line.lower().split(' '):
                wid = 1
                if word in self.input_word_to_index:
                    wid = self.input_word_to_index[word]
                x.append(wid)
                if len(x) >= self.max_length_input:
                    break
            temp.append(x)
        temp = pad_sequences(temp, maxlen=self.max_length_input)

        print(temp.shape)
        return temp

    def load_weights(self, weight_file_path):
        if os.path.exists(weight_file_path):
            self.model.load_weights(weight_file_path)

    def transform_output_encoding(self, texts):
        temp = []
        for line in texts:
            x = []
            line2 = 'START ' + line.lower() + ' END'
            for word in line2.split(' '):
                x.append(word)
                if len(x) >= self.max_length_output:
                    break
            temp.append(x)

        temp = np.array(temp)
        print(temp.shape)
        return temp

    def generate_batch(self, x, y, batch_size):
        num_batches = len(x) // batch_size
        while True:
            for batch_index in range(0, num_batches):
                start = batch_index * batch_size
                end = (batch_index + 1) * batch_size
                encoder_input_data_batch = pad_sequences(x[start:end], self.max_length_input)
                decoder_output_data_batch = np.zeros(
                    shape=(batch_size, self.max_length_output, self.number_output_words))
                for lineIdx, target_words in enumerate(y[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0  # default [UNK]
                        if w in self.output_word_to_index:
                            w2idx = self.output_word_to_index[w]
                        if w2idx != 0:
                            decoder_output_data_batch[lineIdx, idx, w2idx] = 1
                yield encoder_input_data_batch, decoder_output_data_batch

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + '/' + SummarizerRNN.model_name + '-weights.h5'

    @staticmethod
    def get_settings_path(model_dir_path):
        return model_dir_path + '/' + SummarizerRNN.model_name + '-settings.npy'

    @staticmethod
    def get_architecture_path(model_dir_path):
        return model_dir_path + '/' + SummarizerRNN.model_name + '-architecture.json'

    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size, model_dir_path='./models'):
        settings_path = SummarizerRNN.get_settings_path(model_dir_path)
        weight_path = SummarizerRNN.get_weight_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_path)
        np.save(settings_path, self.settings)
        architecture_path = SummarizerRNN.get_architecture_path(model_dir_path)
        open(architecture_path, 'w').write(self.model.to_json())

        y_train = self.transform_output_encoding(y_train)
        y_test = self.transform_output_encoding(y_test)

        x_train = self.transform_input_text(x_train)
        x_test = self.transform_input_text(x_test)

        train_gen = self.generate_batch(x_train, y_train, batch_size)
        test_gen = self.generate_batch(x_test, y_test, batch_size)

        train_num_batches = len(x_train) // batch_size
        test_num_batches = len(x_test) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=1, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1  # default [UNK]
            if word in self.input_word_to_index:
                idx = self.input_word_to_index[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_length_input)
        predicted = self.model.predict(input_seq)
        predicted_word_idx_list = np.argmax(predicted, axis=1)
        predicted_word_list = [self.output_index_to_word[wid] for wid in predicted_word_idx_list[0]]
        return predicted_word_list
