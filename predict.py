import pandas as pd
from rnn import SummarizerRNN
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './corpus'
    model_dir_path = './models'

    print('Загрузка JSONL файла ...')
    df = pd.read_json(data_dir_path + "/test.jsonl", lines=True)[:10]
    # df = df.loc[df.index < 1000]
    X = df['text']
    Y = df['summary']

    config = np.load(SummarizerRNN.get_settings_path(model_dir_path=model_dir_path), allow_pickle=True).item()

    summarizer = SummarizerRNN(config)
    summarizer.load_weights(weight_file_path=SummarizerRNN.get_weight_path(model_dir_path=model_dir_path))

    print('Старт предсказания ...')
    for i in np.random.permutation(np.arange(len(X)))[0:20]:
        x = X[i]
        actual_headline = Y[i]
        headline = summarizer.summarize(x)
        # print('Статья: ', x)
        print('Предсказанный реферат: ', headline)
        print('Настоящий реферат: ', actual_headline)


if __name__ == '__main__':
    main()
