import sklearn as sk

# tf-idf vectorizer
def vectorizer(tokenized_documents):
    ## tokenized_documents is a list of lists, where each inner list contains tokens
    vectorizer = sk.feature_extraction.text.TfidfVectorizer(lowercase=False, preprocessor=None, tokenizer=lambda x: x)
    X = vectorizer.fit_transform(tokenized_documents)
    return X, vectorizer