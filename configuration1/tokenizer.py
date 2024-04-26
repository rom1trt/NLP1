import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import sklearn as sk
import sklearn.model_selection

import tiktoken

nltk.download('punkt')
nltk.download('stopwords')

def tokenizer(X):
    stop_words = set(stopwords.words('english'))
    tokenized_documents = []
    for text in X:
        sentences = sent_tokenize(text)
        tokenized_sentences = [nltk.RegexpTokenizer(r"\w+").tokenize(s) for s in sentences]
        tokenized_document = [word.lower() for sentence in tokenized_sentences for word in sentence if word.lower() not in stop_words]
        tokenized_documents.append(tokenized_document)
    return tokenized_documents