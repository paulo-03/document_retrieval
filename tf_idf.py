"""
This python script implement the class used to use the TF-IDF method,
to retrieve documents from a query.

Author: Paulo Ribeiro
"""

import pandas as pd


class TFIDF:  # Smart move to create a parent class ? Not sure, I will see
    pass


class TFIDFTrainer:
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.corpus = self._load_data()

    def _load_data(self):
        """Load the cleaned data and transform it to have a constant pattern across words,
        meaning no uppercase letter, etc."""
        return pd.read_json(self.corpus_path, lines=True)

    def tf(self):
        """Compute the Term Frequency (TF) of a text"""
        pass

    def idf(self):
        """Compute the Inverse Document Frequency (IDF) of a corpus"""
        pass


class TFIDFRetriever:
    def __init__(self, tf_idf_matrix_path: str):
        self.tf_idf_matrix_path = tf_idf_matrix_path
        self.tf_idf = self._load_tf_idf()

    def _load_tf_idf(self):
        """Load the TF-IDF matrix already computed with TFIDFTrainer method"""
        pass
