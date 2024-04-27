import gensim.downloader as api
from numpy import zeros
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def vectorizer(tokenized_documents):
    model = api.load("word2vec-google-news-300")
    
    def avg_word2vec(tokens):
      missing_words = [token for token in tokens if token not in model]
      word_vectors = [model[token] for token in tokens if token in model]
    
      if missing_words:
          num_dimensions = len(next(iter(word_vectors)))
          zero_vector = zeros(num_dimensions)
          word_vectors.extend([zero_vector] * len(missing_words))
    
      if word_vectors:
          average_vector = sum(word_vectors) / len(word_vectors)
          return average_vector
      else:
          num_dimensions = model.vector_size
          return zeros(num_dimensions)
    
    document_vectors = [avg_word2vec(doc) for doc in tokenized_documents]
    X = np.array(document_vectors)

    scaler = MinMaxScaler()
    scaler.fit(X)
    X_train_scaled = scaler.transform(X)

    return X_train_scaled, None