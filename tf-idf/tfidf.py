from collections import Counter
import math

class TFIDFVectorizer:
    def __init__(self):
        pass

    def fit_transform(self, corpus):
        N = len(corpus)
        term_frequencies = self._compute_term_frequencies(corpus, N)
        tfidf_scores = self._compute_scores(term_frequencies, N)
        return self._to_matrix(tfidf_scores, corpus)
    
    def _compute_term_frequencies(self, corpus, N):
        term_frequencies = {}
        for doc_index, doc in enumerate(corpus):
            term_counts = Counter(doc)
            for term, count in term_counts.items():
                if term not in term_frequencies:
                    term_frequencies[term] = [0] * N
                term_frequencies[term][doc_index] = count
        return term_frequencies

    def _compute_scores(self, term_frequencies, N):
        tfidf_scores = {}
        for term, doc_counts in term_frequencies.items():
            df = sum(1 for count in doc_counts if count > 0)
            idf = math.log(N / df)
            tfidf_scores[term] = [(count * idf if count > 0 else 0) for count in doc_counts]
        return tfidf_scores

    def _to_matrix(self, tfidf_scores, corpus):
        tfidf_matrix = []
        for doc in corpus:
            doc_tfidf = []
            for term, scores in tfidf_scores.items():
                doc_tfidf.append(scores[corpus.index(doc)])
            tfidf_matrix.append(doc_tfidf)
        return tfidf_matrix