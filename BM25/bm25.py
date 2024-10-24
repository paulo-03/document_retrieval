import math
import os
from collections import Counter
from pickle import dump, load, HIGHEST_PROTOCOL
from typing import Union
import concurrent.futures
from collections import Counter, defaultdict

import pandas as pd
from tqdm import tqdm


class BM25_train:
    def __init__(self, corpus_path: str):
        """
        Initializes the BM25 retriever.

        Args:
            corpus_path: Path to the corpus file.
        """
        self.lang = None
        corpus_df = self._load_corpus(corpus_path)

        corpus_text = corpus_df['text'].tolist()
        self.docids = corpus_df['docid'].tolist()
        del corpus_df
        self.corpus_size = len(corpus_text)
        self.doc_len = []

        self.doc_freqs = []  # list of term frequency per document
        self.idf = {}  # inverse document frequency for each term
        self._initialize(corpus_text)
        self.avgdl = sum(self.doc_len) / self.corpus_size

    def _load_corpus(self, corpus_path):
        """Load the cleaned corpus, meaning a transformed documents to have a constant pattern across words,
        meaning no uppercase letter, etc."""
        # Load the data
        print("Loading corpus...")
        corpus = pd.read_json(corpus_path, lines=True)
        print("Corpus loaded successfully !\n")

        # Give some basic information about the data
        self.lang = corpus['lang'].unique()
        print("Information about the given corpus\n"
              "###################################\n"
              f"Number of documents: {len(corpus)}\n"
              f"Language (only one language should be displayed): {self.lang}\n")

        return corpus

    def _initialize(self, corpus_text):
        """Precompute document frequencies and IDF for all terms in the corpus."""
        # Compute document frequencies
        term_doc_count = Counter()

        for doc in tqdm(corpus_text, desc="Computing TFs"):

            corpus_tokenized = doc.split()
            self.doc_len.append(len(corpus_tokenized))
            doc_term_count = Counter(corpus_tokenized)
            self.doc_freqs.append(doc_term_count)
            for term in doc_term_count.keys():
                term_doc_count[term] += 1

        # Compute IDF for each term
        for term, doc_freq in tqdm(term_doc_count.items(), desc="Computing IDF"):
            # Apply smoothing to avoid division by zero
            self.idf[term] = math.log(self.corpus_size + 1) - math.log(doc_freq)


    def save_pickle(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        results = {
            'lang': self.lang,
            'docids': self.docids,
            'idf': self.idf,
            'doc_freqs': self.doc_freqs,
            'corpus_size': self.corpus_size,
            'doc_len': self.doc_len,
            'avgdl': self.avgdl
        }
        with open(path, 'wb') as f:
            dump(results, f, protocol=HIGHEST_PROTOCOL)


class BM25_retriever:
    def     __init__(self, train_path: str, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        """
        Initializes the BM25 retriever.

        Args:
            train_path: Path to the trained BM25 model.
            k1: Hyperparameter for term frequency scaling.
            b: Hyperparameter for length normalization.
            delta: Hyperparameter for smoothing.
        """

        self.k1 = k1
        self.b = b
        self.delta = delta
        self.lang = None
        self.docids = None
        self.corpus_size = None
        self.doc_len = None
        self.avgdl = None
        self.idf = None
        self.doc_freqs = None

        self._load_model(train_path)

    def _load_queries(self, query_path: str):
        """Load the cleaned data, meaning a transformed documents to have a constant pattern across words,
        meaning no uppercase letter, etc."""
        # Load the data
        print("Loading queries...")
        queries = pd.read_csv(query_path)
        queries = queries[queries['lang'] == self.lang]
        print("Queries loaded successfully !\n")

        print("Information about the given queries\n"
              "###################################\n"
              f"Number of queries: {len(queries)}\n"
              f"Language (only one language should be displayed): {queries['lang'].unique()}\n")

        return queries

    def _score(self, query, document_idx):
        """
        Computes the BM25 score for a document given a query.
        :param query: List of query terms
        :param document_idx: Index of the document in the corpus
        :return: BM25 score for the document
        """
        doc_term_count = self.doc_freqs[document_idx]
        doc_length = self.doc_len[document_idx]
        score = 0.0

        for term in query:
            if term not in doc_term_count:
                continue  # term not in document, skip

            term_freq = doc_term_count[term]
            idf = self.idf.get(term, 0)  # Get IDF, 0 if term not in corpus
            numerator = term_freq + self.delta * term_freq
            denominator = term_freq + self.delta + self.k1 * (1 - self.b + self.b * (doc_length / self.avgdl))
            score += idf * (numerator / denominator)

        return score

    def _rank(self, query):
        """
        Ranks all documents in the corpus for a given query.
        :param query: List of query terms
        :return: List of (document_idx, score) tuples sorted by score
        """
        scores = [(idx, self._score(query, idx)) for idx in range(self.corpus_size)]
        return sorted(scores, key=lambda x: x[1], reverse=True)

    def top_10_docid_for_all_queries(self, query_source: Union[str, pd.DataFrame]) -> dict:
        """
        Retrieve the top 10 documents for each query and return them in a dictionary.
        :param query_source: Path to the query file or a DataFrame containing the queries.
        :return: Dictionary mapping query_id to a list of top 10 docids
        """
        if isinstance(query_source, str):
            query_df = self._load_queries(query_source)
        else:
            query_df = query_source

        queries = query_df['text'].tolist()
        queries_ids = query_df['query_id'].tolist()

        top_10_docids = {}
        for query_id, query in tqdm(zip(queries_ids, queries), desc="Ranking queries",
                                    total=len(queries_ids)):
            query = query.split()
            ranked = self._rank(query)
            top_10_docids[query_id] = [self.docids[idx] for idx, _ in ranked[:10]]

        return top_10_docids

    def _load_model(self, train_path: str):
        with open(train_path, 'rb') as f:
            train_results = load(f)
            self.lang = train_results['lang']
            self.docids = train_results['docids']
            self.idf = train_results['idf']
            self.doc_freqs = train_results['doc_freqs']
            self.corpus_size = train_results['corpus_size']
            self.doc_len = train_results['doc_len']
            self.avgdl = train_results['avgdl']
