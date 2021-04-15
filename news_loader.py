from collections import Counter

MAX_INPUT_SEQ_LENGTH = 500
MAX_TARGET_SEQ_LENGTH = 50
MAX_INPUT_VOCAB_SIZE = 5000
MAX_TARGET_VOCAB_SIZE = 2000


def fit_text(x, y, input_seq_max_length=MAX_INPUT_SEQ_LENGTH, output_seq_max_length=MAX_TARGET_SEQ_LENGTH):
    input_counter = Counter()
    output_counter = Counter()
    max_length_input = 0
    max_length_output = 0

    for line in x:
        text = [word.lower() for word in line.split(' ')]
        seq_length = len(text)
        if seq_length > input_seq_max_length:
            text = text[0:input_seq_max_length]
            seq_length = len(text)
        for word in text:
            input_counter[word] += 1
        max_length_input = max(max_length_input, seq_length)

    for line in y:
        line2 = 'START ' + line.lower() + ' END'
        text = [word for word in line2.split(' ')]
        seq_length = len(text)
        if seq_length > output_seq_max_length:
            text = text[0:output_seq_max_length]
            seq_length = len(text)
        for word in text:
            output_counter[word] += 1
            max_length_output = max(max_length_output, seq_length)

    input_word_to_index = dict()
    for idx, word in enumerate(input_counter.most_common(MAX_INPUT_VOCAB_SIZE)):
        input_word_to_index[word[0]] = idx + 2
    input_word_to_index['PAD'] = 0
    input_word_to_index['UNK'] = 1
    input_index_to_word = dict([(idx, word) for word, idx in input_word_to_index.items()])

    output_word_to_index = dict()
    for idx, word in enumerate(output_counter.most_common(MAX_TARGET_VOCAB_SIZE)):
        output_word_to_index[word[0]] = idx + 1
    output_word_to_index['UNK'] = 0

    output_index_to_word = dict([(idx, word) for word, idx in output_word_to_index.items()])

    number_input_words = len(input_word_to_index)
    number_output_words = len(output_word_to_index)

    settings = dict()
    settings['input_word_to_index'] = input_word_to_index
    settings['input_index_to_word'] = input_index_to_word
    settings['output_word_to_index'] = output_word_to_index
    settings['output_index_to_word'] = output_index_to_word
    settings['number_input_words'] = number_input_words
    settings['number_output_words'] = number_output_words
    settings['max_length_input'] = max_length_input
    settings['max_length_output'] = max_length_output

    return settings