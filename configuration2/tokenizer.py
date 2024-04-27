import pandas as pd
import tiktoken
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def tokenizer(X):
    enc = tiktoken.encoding_for_model("gpt-4")
    stop_words = set(stopwords.words('english'))
    num_tokens = 0
    num_sentences = 0
    tokenized_documents = []
    for text in X:
        tokenized_document = enc.encode(text)

        filtered_tokens = [token for token in tokenized_document if token not in stop_words]

        num_tokens += len(filtered_tokens)
        num_sentences += text.count('.') + text.count('!') + text.count('?')
        
        tokenized_documents.append(filtered_tokens)

    print("Number of tokens: ", num_tokens)
    print("Number of sentences: ", num_sentences)
    
    return tokenized_documents
