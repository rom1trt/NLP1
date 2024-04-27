import pandas as pd
import spacy

def tokenizer(X):
    nlp = spacy.load("en_core_web_sm")

    num_tokens = 0
    num_sentences = 0
    tokenized_data = []
    for doc in nlp.pipe(X, batch_size=5, disable=["parser", "ner"]):
        tokens = [token.text for token in doc]
        num_tokens += len(tokens)
        num_sentences += len(list(doc.sents))
        tokenized_data.append(tokens)
    
    print("Number of tokens: ", num_tokens)
    print("Number of sentences: ", num_sentences)
    
    return tokenized_data
