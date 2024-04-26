import pandas as pd
import spacy

def tokenizer(X):
    nlp = spacy.load("en_core_web_sm")

    tokenized_data = []
    for doc in nlp.pipe(X, batch_size=5, disable=["parser", "ner"]):
      tokens = [token.text for token in doc]
      tokenized_data.append(tokens)
        
    return tokenized_data

