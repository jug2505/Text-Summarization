from attention import AttentionLayer

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import warnings

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")

data_train = pd.read_json("corpus/train.jsonl", lines=True)
data_val = pd.read_json("corpus/val.jsonl", lines=True)
data_test = pd.read_json("corpus/test.jsonl", lines=True)

data_train.info()
data_val.info()
data_test.info()

