import pandas as pd
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import sklearn as sk
import sklearn.model_selection

import tiktoken

nltk.download('punkt')
nltk.download('stopwords')

def tokenizer(X):
    stop_words = set(stopwords.words('english'))
    num_tokens = 0
    num_sentences = 0
    tokenized_documents = []
    lemmatizer = WordNetLemmatizer()
    for text in X:
        sentences = sent_tokenize(text)
        num_sentences += len(sentences)
        tokenized_sentences = [nltk.RegexpTokenizer(r"\w+").tokenize(s) for s in sentences]
        tokenized_document = [lemmatizer.lemmatize(word.lower()) for sentence in tokenized_sentences for word in sentence if word.lower() not in stop_words]
        num_tokens += len(tokenized_document)
        tokenized_documents.append(tokenized_document)
    print("Number of tokens: ", num_tokens)
    print("Number of sentences: ", num_sentences)
    return tokenized_documents
