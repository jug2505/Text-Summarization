from nltk.corpus import stopwords
import numpy as np
from nltk.cluster.util import cosine_distance
import pandas as pd
import networkx as nx


def similarity(sentence_1, sentence_2, stop_words):
    """
    Вычисление косуного расстояния между 2 предложениями
    """
    sentence_1 = [word.lower() for word in sentence_1.split(' ')]
    sentence_2 = [word.lower() for word in sentence_2.split(' ')]
    words_of_sentences = list(set(sentence_1 + sentence_2))
    vector_1_sent = [0] * len(words_of_sentences)
    vector_2_sent = [0] * len(words_of_sentences)
    # Построение вектора для первого предложения
    for word in sentence_1:
        if word in stop_words:
            continue
        vector_1_sent[words_of_sentences.index(word)] += 1
    # Построение вектора для второго предложения
    for word in sentence_2:
        if word in stop_words:
            continue
        vector_2_sent[words_of_sentences.index(word)] += 1

    return 1 - cosine_distance(vector_1_sent, vector_2_sent)


def compute_sim_matrix(sentences, stop_words):
    """
    Вычисление схожести между всеми предложениями
    """
    sent_len = len(sentences)
    similarity_matrix = np.zeros((sent_len, sent_len))
    for i in range(sent_len):
        for j in range(sent_len):
            if i == j:  # Если это одно и тоже предложение
                continue
            similarity_matrix[i][j] = similarity(sentences[i], sentences[j], stop_words)

    return similarity_matrix


def summary(file_path, number, num_of_sentences_result=2):
    stop_words = stopwords.words('russian')
    # Чтение текста
    df = pd.read_json(file_path, lines=True)
    sentences = df['text'][number].split('. ')
    # Вычисление матрицы схожести
    sim_matrix = compute_sim_matrix(sentences, stop_words)
    # Применение textrank
    sim_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(sim_graph)
    # Сортировка и выбор верхних предложений
    sent_ranked = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    print("Индексы наиболее высоко ранжированных предложений ", sent_ranked)
    summarize_text = []
    for i in range(num_of_sentences_result):
        summarize_text.append(sent_ranked[i][1])
    print("Реферат: ", ". ".join(summarize_text))


def main():
    summary("corpus/test.jsonl", 0, 2)


if __name__ == '__main__':
    main()
