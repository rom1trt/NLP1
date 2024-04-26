import pandas as pd
import tiktoken

def tokenizer(X):
    X_strings = " ".join(X)
    enc = tiktoken.encoding_for_model("gpt-4")
    enc.encode("hello world")
    X_tokens = enc.encode(X_strings)
    return X_tokens