import pandas as pd
import tiktoken

def tokenizer(X):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenized_documents = []
    for text in X:
        tokenized_document = enc.encode(text)
        tokenized_documents.append(tokenized_document)
    return tokenized_documents