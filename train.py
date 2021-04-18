import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from seq2seq import Summarizer
from news_loader import fit_text
import numpy as np


def create_history_plot(history, model_name):
    plt.title(model_name)
    plt.plot(history.history['loss'], color='r', label='Train')
    plt.plot(history.history['val_loss'], color='m', label='Validation')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(42)
    data_dir_path = './corpus'

    print('Загрузка jsonl файла ...')
    df = pd.read_json(data_dir_path + "/train.jsonl", lines=True)

    y = df['summary']
    x = df['text']

    print('Создание файла настройки ...')
    settings = fit_text(x, y)
    print('Файл настройки создан ...')

    summarizer = Summarizer(settings)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print('Размер тренировочных данных: ', len(x_train))
    print('Размер тестовых данных: ', len(x_test))

    print('Начало обучения ...')
    history = summarizer.fit(x_train, y_train, x_test, y_test, epochs=10, batch_size=20)

    create_history_plot(history, summarizer.model_name)


if __name__ == '__main__':
    main()
