"""
Python script for the BM25 implementation.
"""

import pandas as pd


class BM25Vectorizer:
    def __init__(self):
        pass

    def _load_corpus(self):
        pass


class BM25Retriever(BM25Vectorizer):
    def __init__(self, vectorized_corpus_path: str, clean_query_df: pd.DataFrame):
        super().__init__()
        pass

    def _load_vectorized_corpus(self):
        pass
