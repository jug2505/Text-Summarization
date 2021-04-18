from keras.models import Model
from keras.layers import Embedding, Dense, Input
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

HIDDEN_UNITS = 500
VERBOSE = 1


class Summarizer(object):

    model_name = 'seq2seq'

    def __init__(self, settings):
        self.number_input_words = settings['number_input_words']
        self.max_length_input = settings['max_length_input']
        self.number_output_words = settings['number_output_words']
        self.max_length_output = settings['max_length_output']
        self.input_word_to_index = settings['input_word_to_index']
        self.input_index_to_word = settings['input_index_to_word']
        self.output_word_to_index = settings['output_word_to_index']
        self.output_index_to_word = settings['output_index_to_word']
        self.config = settings

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(input_dim=self.number_input_words, output_dim=HIDDEN_UNITS,
                                      input_length=self.max_length_input, name='encoder_embedding')
        encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
        encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_embedding(encoder_inputs))
        encoder_states = [encoder_state_h, encoder_state_c]

        decoder_inputs = Input(shape=(None, self.number_output_words), name='decoder_inputs')
        decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
        decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                                         initial_state=encoder_states)
        decoder_dense = Dense(units=self.number_output_words, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        self.model = model

        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

    def load_weights(self, weight_path):
        if os.path.exists(weight_path):
            self.model.load_weights(weight_path)

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

    def transform_target_encoding(self, texts):
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
                decoder_output_data_batch = np.zeros(shape=(batch_size, self.max_length_output, self.number_output_words))
                decoder_input_data_batch = np.zeros(shape=(batch_size, self.max_length_output, self.number_output_words))
                for lineIdx, target_words in enumerate(y[start:end]):
                    for idx, w in enumerate(target_words):
                        w2idx = 0
                        if w in self.output_word_to_index:
                            w2idx = self.output_word_to_index[w]
                        if w2idx != 0:
                            decoder_input_data_batch[lineIdx, idx, w2idx] = 1
                            if idx > 0:
                                decoder_output_data_batch[lineIdx, idx - 1, w2idx] = 1
                yield [encoder_input_data_batch, decoder_input_data_batch], decoder_output_data_batch

    @staticmethod
    def get_weight_path(model_dir_path):
        return model_dir_path + '/' + Summarizer.model_name + '-weights.h5'

    @staticmethod
    def get_config_path(model_dir_path):
        return model_dir_path + '/' + Summarizer.model_name + '-settings.npy'

    @staticmethod
    def get_architecture_path(model_dir_path):
        return model_dir_path + '/' + Summarizer.model_name + '-architecture.json'

    def fit(self, x_tr, y_tr, x_test, y_test, epochs=None, batch_size=None, model_dir_path='./models'):
        config_file_path = Summarizer.get_config_path(model_dir_path)
        weight_file_path = Summarizer.get_weight_path(model_dir_path)
        checkpoint = ModelCheckpoint(weight_file_path)
        np.save(config_file_path, self.config)
        architecture_file_path = Summarizer.get_architecture_path(model_dir_path)
        open(architecture_file_path, 'w').write(self.model.to_json())

        y_tr = self.transform_target_encoding(y_tr)
        y_test = self.transform_target_encoding(y_test)

        x_tr = self.transform_input_text(x_tr)
        x_test = self.transform_input_text(x_test)

        train_gen = self.generate_batch(x_tr, y_tr, batch_size)
        test_gen = self.generate_batch(x_test, y_test, batch_size)

        train_num_batches = len(x_tr) // batch_size
        test_num_batches = len(x_test) // batch_size

        history = self.model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                                           epochs=epochs,
                                           verbose=VERBOSE, validation_data=test_gen, validation_steps=test_num_batches,
                                           callbacks=[checkpoint])
        self.model.save_weights(weight_file_path)
        return history

    def summarize(self, input_text):
        input_seq = []
        input_wids = []
        for word in input_text.lower().split(' '):
            idx = 1
            if word in self.input_word_to_index:
                idx = self.input_word_to_index[word]
            input_wids.append(idx)
        input_seq.append(input_wids)
        input_seq = pad_sequences(input_seq, self.max_length_input)
        states_value = self.encoder_model.predict(input_seq)
        output_seq = np.zeros((1, 1, self.number_output_words))
        output_seq[0, 0, self.output_word_to_index['START']] = 1
        target_text = ''
        target_text_len = 0
        terminated = False
        while not terminated:
            output_tokens, h, c = self.decoder_model.predict([output_seq] + states_value)

            sample_token_idx = np.argmax(output_tokens[0, -1, :])
            sample_word = self.output_index_to_word[sample_token_idx]
            target_text_len += 1

            if sample_word != 'START' and sample_word != 'END':
                target_text += ' ' + sample_word

            if sample_word == 'END' or target_text_len >= self.max_length_output:
                terminated = True

            output_seq = np.zeros((1, 1, self.number_output_words))
            output_seq[0, 0, sample_token_idx] = 1

            states_value = [h, c]
        return target_text.strip()
