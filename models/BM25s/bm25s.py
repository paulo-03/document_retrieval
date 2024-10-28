"""
This python script implement the class used to use the BM25s method, inspired by TF-IDF
to retrieve documents from a query.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import sparse
from collections import Counter
from tqdm import tqdm
from pickle import load, dump, HIGHEST_PROTOCOL
from multiprocessing import Pool, cpu_count


def tokenize(doc):
    """Efficient tokenization using regex."""
    return re.findall(r'\b\w+\b', doc)


class BM25sTrainer:
    def __init__(self, corpus_path: str, k1: float, b: float):
        self.lang = None
        self.data = self._load_data(corpus_path)
        self.vocabulary = None
        self.vocab_len = None
        self.avgdl = None
        self.doc_lengths = None
        self.norm_tf_matrix = None
        self.idf = None
        self.score_matrix = None
        self.k1 = k1
        self.b = b

    def _load_data(self, corpus_path):
        """Load data and look the language of the data."""
        print("Loading corpus...")
        corpus = pd.read_json(corpus_path, lines=True)
        lang = corpus['lang'].unique()
        if len(lang) == 1:
            self.lang = lang[0]
            print(f"Corpus loaded successfully! Found {len(corpus)} documents in '{self.lang}'.\n")
            return corpus
        else:
            raise Warning(f"The corpus you are giving is multilingual ({len(lang)} different languages)."
                          "Please give only a corpus with one languages.")

    def create_vocabulary(self):
        """Create the vocabulary, document lengths and average document length based on the document corpus."""
        word_count = Counter()
        doc_length = []

        print("Starting computing corpus vocabulary...")
        for doc in self.data['text']:
            tokens = tokenize(doc)
            word_count.update(tokens)
            doc_length.append(len(tokens))

        self.vocabulary = sorted(word_count.keys())
        self.vocab_len = len(self.vocabulary)
        self.doc_lengths = np.array(doc_length)
        self.avgdl = np.mean(self.doc_lengths)
        print(f"Vocabulary created with {self.vocab_len} unique terms!\n"
              f"Computed document lengths and average document length: {self.avgdl:.2f} tokens/words\n")

    def _compute_norm_tf(self):
        """Compute term frequency (TF) matrix for the queries."""
        doc_ids = []
        token_ids = []
        tf_values = []
        token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        doc_number = len(self.data)

        for doc_id, doc in tqdm(enumerate(self.data['text']), total=doc_number, desc=f'Computing normalized TF'):
            tokens = tokenize(doc)
            token_freq = Counter(tokens)
            doc_length = self.doc_lengths[doc_id]

            for token, freq in token_freq.items():
                if token in token_to_id:
                    # Compute the normalized TF from BM25s score
                    nominator = freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + self.b * doc_length / self.avgdl)
                    value = nominator / denominator
                    # Store the values to create a sparse matrix
                    doc_ids.append(doc_id)
                    token_ids.append(token_to_id[token])
                    tf_values.append(value)

        return sparse.csr_matrix((tf_values, (doc_ids, token_ids)), shape=(doc_number, self.vocab_len))

    def _compute_idf(self):
        """Compute the inverse document frequency (IDF) for the corpus."""
        N = len(self.data)
        df = self.norm_tf_matrix.astype(bool).sum(axis=0)
        return sparse.csr_matrix(np.log((N + 1) / (df + 1)) + 1)

    def fit(self):
        """Fit the BM25S model to the corpus by computing necessary statistics."""
        print(f"Initiate BM25s model training for '{self.lang}' documents.")
        self.norm_tf_matrix = self._compute_norm_tf()
        print("Computing IDF...")
        self.idf = self._compute_idf()
        print("Computing score matrix (IDF x normalized TF)...")
        self.score_matrix = sparse.csr_matrix(self.norm_tf_matrix.multiply(self.idf))
        print(f"BM25s model is ready for {self.lang} document retrieval!\n")

    def save_results(self, path: str):
        """Save the computed data to a file for future use."""
        os.makedirs(f'{path}/k1_{self.k1}', exist_ok=True)
        results = {
            'vocabulary': self.vocabulary,
            'vocab_len': self.vocab_len,
            'id_to_docid': {idx: docid for idx, docid in enumerate(self.data['docid'])},
            'score_matrix': self.score_matrix,
            'lang': self.lang,
            'k1': self.k1,
            'b': self.b
        }

        with open(f'{path}/k1_{self.k1}/bm25s_{self.lang}.pkl', 'wb') as f:
            dump(results, f, protocol=HIGHEST_PROTOCOL)
        print(f"BM25s model for '{self.lang}' saved successfully! (k1: {self.k1}, b: {self.b})\n\n")


class BM25sRetriever:
    def __init__(self, queries_df: pd.DataFrame, model_path: str, top_k: int = 10):
        # Load the precomputed BM25S model
        with open(model_path, 'rb') as f:
            model_data = load(f)
        # class variable
        self.vocabulary = model_data['vocabulary']
        self.vocab_len = model_data['vocab_len']
        self.id_to_docid = model_data['id_to_docid']
        self.score_matrix = model_data['score_matrix']
        self.lang = model_data['lang']

        # Initialize query data
        self.data = queries_df
        self.query_tf_matrix = None
        self.matches = None
        self.top_k = top_k

        # Check and information variable
        self.check_info(model_lang=self.lang, queries_lang=self.data['lang'].unique()[0],
                        k1=model_data['k1'], b=model_data['b'])
        del model_data

    @staticmethod
    def check_info(model_lang: str, queries_lang: str, k1: float, b: float):
        """TODO"""
        if model_lang != queries_lang:
            error_msg = (f"Model language ({model_lang}) and queries language ({queries_lang}) are not equivalent. "
                         f"Please make sure to use the right model for your current queries.")
            raise ValueError(error_msg)
        else:
            print(f"BM25s retriever for '{model_lang}' queries ready to go ! (k1: {k1}, b: {b})\n")

    def _compute_tf(self):
        """Compute term frequency (TF) matrix for the queries."""
        query_ids = []
        token_ids = []
        tf_values = []
        token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        queries_number = len(self.data)

        for doc_id, doc in tqdm(enumerate(self.data['text']),
                                total=queries_number,
                                desc=f"Computing queries TF ('{self.lang}')"):
            tokens = tokenize(doc)
            token_freq = Counter(tokens)

            for token, freq in token_freq.items():
                if token in token_to_id:
                    # Store the values to create a sparse matrix
                    query_ids.append(doc_id)
                    token_ids.append(token_to_id[token])
                    tf_values.append(freq)

        return sparse.csr_matrix((tf_values, (query_ids, token_ids)), shape=(queries_number, self.vocab_len))

    def _score_single_query(self, query_doc_scores):
        """Compute BM25S score for a single query against all documents and select the top k docs."""
        query_bm25_scores = query_doc_scores.toarray().flatten()
        return [self.id_to_docid[idx] for idx in np.argsort(query_bm25_scores)[-self.top_k:][::-1]]

    def match(self):
        """Perform document retrieval for the queries."""
        # Start by computing the queries TF to know the occurrence of each word in the query to compute the score.
        self.query_tf_matrix = self._compute_tf()

        # Compute the query x doc score matrix
        queries_docs_scores = self.query_tf_matrix.dot(self.score_matrix.T)

        # Convert query TF matrix rows to a list of sparse matrix rows
        queries_docs_scores = [queries_docs_scores.getrow(i) for i in range(queries_docs_scores.shape[0])]

        print(f"Computing BM25S scores with multiprocessing for '{self.lang}' queries...")
        with Pool(processes=int(4)) as pool:
            # Pass the list of sparse rows to the pool and compute top-k results for each query
            top_k_docs = pool.map(self._score_single_query, queries_docs_scores)

        # Save matches as a pandas Series
        self.matches = pd.Series(top_k_docs, name='docids')
        print(f"Retrieved top-{self.top_k} documents for each '{self.lang}' query.")

    def show_results(self):
        """Display results."""
        print(self.matches)
