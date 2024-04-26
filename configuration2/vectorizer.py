import sklearn.feature_extraction.text as sk_text

def vectorizer(tokenized_documents):
    # Conversion des bytes en chaînes de caractères si nécessaire
    tokenized_documents_str = []
    for document in tokenized_documents:
        if isinstance(document, list):
            # Convertir les bytes en chaînes de caractères
            tokenized_document_str = [token.decode('utf-8') if isinstance(token, bytes) else str(token) for token in document]
            tokenized_documents_str.append(tokenized_document_str)
        else:
            # Si le document n'est pas une liste, le convertir en liste de chaîne de caractères
            tokenized_documents_str.append([str(document)])

    # Initialiser le vectoriseur
    vectorizer = sk_text.CountVectorizer(lowercase=False, preprocessor=None, tokenizer=lambda x: x)
    
    # Adapter et transformer les données
    X = vectorizer.fit_transform(tokenized_documents_str)
    
    return X, vectorizer
