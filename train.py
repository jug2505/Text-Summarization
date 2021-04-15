import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from rnn import SummarizerRNN
from news_loader import fit_text
import numpy as np


def create_history_plot(history, model_name, metrics=None):
    plt.title(model_name)
    if metrics is None:
        metrics = {'acc', 'loss'}
    if 'acc' in metrics:
        plt.plot(history.history['acc'], color='g', label='Train Accuracy')
        plt.plot(history.history['val_acc'], color='b', label='Validation Accuracy')
    if 'loss' in metrics:
        plt.plot(history.history['loss'], color='r', label='Train Loss')
        plt.plot(history.history['val_loss'], color='m', label='Validation Loss')
    plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(42)
    data_dir_path = './corpus'

    print('Загрузка jsonl файла ...')
    df = pd.read_json(data_dir_path + "/train.jsonl", lines=True)[:1000]

    y = df['summary']
    x = df['text']

    print('Создание файла настройки ...')
    settings = fit_text(x, y)
    print('Файл настройки создан ...')

    summarizer = SummarizerRNN(settings)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print('Размер тренировочных данных: ', len(x_train))
    print('Размер тестовых данных: ', len(x_test))

    print('Начало обучения ...')
    history = summarizer.fit(x_train, y_train, x_test, y_test, epochs=100, batch_size=20)

    create_history_plot(history, summarizer.model_name, metrics={'loss'})


if __name__ == '__main__':
    main()