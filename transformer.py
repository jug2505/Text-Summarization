# transformer.py
# Класс нормализации текста
# Используется стеммер Портера (Snowball)

import nltk
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer

import nltk
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer


# Класс для нормализации текста
class TextNormalizer:

    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('russian'))

    @staticmethod
    def is_punct(token):
        """
        Сравнивает первую букву в названии категориии Юникода каждого
        символа c P (Punctuation)
        """
        return all(unicodedata.category(char).startswith('P') for char in token)

    def is_stopword(self, token):
        """
        Является ли токен стоп-словом
        """
        return token.lower() in self.stopwords


    def normalize(self, sent):
        """
        Нормализация
        """
        return [
            token.lower()
            for token in sent
            if not self.is_stopword(token) and not self.is_punct(token)
        ]

    # fit и transform для pipeline
    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        norm_corp = []
        for comment in documents:
            norm_corp.append(self.normalize(comment))
        return norm_corp
