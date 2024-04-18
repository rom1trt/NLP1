import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import sklearn as sk
import sklearn.model_selection

nltk.download('punkt')

def tokenizer(X):
    tokenized_documents = []
    for text in X:
        sentences = sent_tokenize(text)
        tokenized_sentences = [nltk.RegexpTokenizer(r"\w+").tokenize(s) for s in sentences]
        tokenized_document = [word.lower() for sentence in tokenized_sentences for word in sentence]
        tokenized_documents.append(tokenized_document)
    return tokenized_documents