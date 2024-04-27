import pandas as pd
import tiktoken

def tokenizer(X):
    enc = tiktoken.encoding_for_model("gpt-4")
    num_tokens = 0
    num_sentences = 0
    tokenized_documents = []
    for text in X:
        tokenized_document = enc.encode(text)
        num_tokens += len(tokenized_document)
        num_sentences += text.count('.') + text.count('!') + text.count('?')
        tokenized_documents.append(tokenized_document)
    print("Number of tokens: ", num_tokens)
    print("Number of sentences: ", num_sentences)
    return tokenized_documents
